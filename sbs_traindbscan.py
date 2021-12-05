from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import hdbscan
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy import sparse as sp
import collections
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
from torch.nn import Parameter
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from UDAsbs import datasets
from UDAsbs import models
from UDAsbs.trainers import DbscanBaseTrainer
from UDAsbs.evaluators import Evaluator, extract_features
from UDAsbs.utils.data import IterLoader
from UDAsbs.utils.data import transforms as T
from UDAsbs.utils.data.sampler import RandomMultipleGallerySampler
from UDAsbs.utils.data.preprocessor import Preprocessor
from UDAsbs.utils.logging import Logger
from UDAsbs.utils.serialization import load_checkpoint, save_checkpoint  # , copy_state_dict
from UDAsbs.utils.faiss_rerank import compute_jaccard_distance

start_epoch = best_mAP = 0


def get_data(name, data_dir, l=1):
    root = osp.join(data_dir)

    dataset = datasets.create(name, root, l)
    ground_truth_label=[]
    label_dict = {}
    for i, item_l in enumerate(dataset.train):
        ground_truth_label.append(item_l[1])
        if item_l[1] in label_dict:
            label_dict[item_l[1]].append(i)
        else:
            label_dict[item_l[1]] = [i]
    return dataset, label_dict,ground_truth_label

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_train_loader(dataset, height, width, choice_c, batch_size, workers,
                     num_instances, iters, trainset=None):
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.596, 0.558, 0.497])
    ])

    train_set = trainset  # dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances, choice_c)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                     transform=train_transformer),
                   batch_size=batch_size, num_workers=0, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def create_model(args, ncs, wopre=False):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                            num_classes=ncs)

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                                num_classes=ncs)

    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model_1)
    copy_state_dict(initial_weights['state_dict'], model_1_ema)

    print('load pretrain model:{}'.format(args.init_1))

    model_1.cuda()
    model_1_ema.cuda()

    model_1 = nn.DataParallel(model_1)
    model_1_ema = nn.DataParallel(model_1_ema)

    for i, cl in enumerate(ncs):
        exec('model_1_ema.module.classifier{}_{}.weight.data.copy_(model_1.module.classifier{}_{}.weight.data)'.format(i,cl,i,cl))
    return model_1, None, model_1_ema, None  # model_1, model_2, model_1_ema, model_2_ema

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)



def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def compute_dist(inputs):
    n = inputs.size(0)
    dist = torch.pow(inputs.cpu(), 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs.cpu(), inputs.cpu().t())
    return dist

def get_uncer_id_by_sc_score(pseudo_labels,sc_score_sample,tao=0.0):
    print('reCluster!use silhouette score and tao={}'.format(tao))
    uncer_id = []
    uncer_num = []
    cer_num = []
    uncer_digital = []
    cer_digital = []
    for id in list(set(pseudo_labels)):
        if id != -1:
            index = (pseudo_labels == id).nonzero()
            class_sc = np.mean(sc_score_sample[index])
            if class_sc <= tao:
                uncer_id.append(id)
                uncer_num.append(len(index[0]))
                uncer_digital.append(class_sc)
            else:
                cer_num.append(len(index[0]))
                cer_digital.append(class_sc)
    print('each uncer num=', uncer_num, 'average=', np.mean(uncer_num), 'uncer_average=', np.mean(uncer_digital))
    return uncer_id

def refine_label(label, max_class,count):
    for ith in range(len(label)):
        if label[ith] != -1:
            label[ith] = label[ith] + max_class + count * 100
    return label

def reCluster(target_features,pseudo_labels,uncer_id, max_class,args,sc_score_sample):
    count = 1
    temp_labels = pseudo_labels.copy()
    for id in list(set(uncer_id)):
        re_cluster = DBSCAN(eps=0.4, min_samples=4, metric='precomputed', n_jobs=-1)
        uncer_cluster_index = (pseudo_labels == id).nonzero()
        feature_recluster = target_features[uncer_cluster_index].squeeze(1)
        rerank_dist_1 = compute_jaccard_distance(feature_recluster, k1=args.k1, k2=args.k2,print_flag=False)
        re_pseudo_labels = re_cluster.fit_predict(rerank_dist_1)
        temp_labels[uncer_cluster_index] = refine_label(re_pseudo_labels, max_class, count)
        # print('the uncer_id {} (include {} samples) re-cluster into {} new clusters,which is id {}'.format(id,len(uncer_cluster_index[0]),len(set(re_pseudo_labels)),set(re_pseudo_labels)))
        count += 1
    return temp_labels

def get_features(model, loader):
    target_features_dict, _ = extract_features(model, loader, print_freq=100)
    target_features = torch.stack(list(target_features_dict.values()))
    target_features = F.normalize(target_features, dim=1)
    return target_features

def get_cluster_item(cluster_centers):
    cluster_features = []
    c_labels = []
    for key in sorted(cluster_centers.keys()):
        cluster_features.append(F.normalize(torch.stack(cluster_centers[key]).mean(0, keepdim=True)))
        c_labels.append(key)
    return cluster_features, c_labels

# generate clean and noise samples
def generate_pseudo_labels(cluster_id, num,dataset_target):
    labels = []
    outliers = 0
    outlier_label = torch.zeros(len(dataset_target.train), len(dataset_target.train))
    for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
        if id != -1:
            labels.append(id)
        else:
            outlier_label[i, i] = 1
            labels.append(num + outliers)
            outliers += 1
    return torch.Tensor(labels).long(), outlier_label

def main_worker(args):

    global start_epoch, best_mAP
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    # Create data loaders
    iters = args.iters if (args.iters > 0) else None
    ncs = [int(x) for x in args.ncs.split(',')]
    dataset_target, label_dict,ground_label_list = get_data(args.dataset_target, args.data_dir, len(ncs))
    dataset_source, _ ,_= get_data(args.dataset_source, args.data_dir, len(ncs))
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    tar_cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,testset=dataset_target.train)
    # Create model
    fc_len = 32621
    model, _, model_ema, _ = create_model(args, [fc_len for _ in range(len(ncs))])
    epoch = 0
    eps = 0.6
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    # cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    cluster = hdbscan.HDBSCAN(metric='precomputed')
    evaluator = Evaluator(model)
    evaluator_ema = Evaluator(model_ema)
    clusters = [args.num_clusters] * args.epochs  # TODO: dropout clusters

    print("Training begining~~~~~~!!!!!!!!!")
    for epoch in range(len(clusters)):
        iters_ = 300 if epoch % 1 == 0 else iters
        if epoch % 1 == 0:
            print('==> Create pseudo labels for unlabeled target domain with model')
            target_features = get_features(model,tar_cluster_loader)
            rerank_dist = compute_jaccard_distance(target_features, k1=args.k1, k2=args.k2)
            pseudo_labels = cluster.fit_predict(rerank_dist.astype('double'))#numbel label
            pl = pseudo_labels.copy()
            num_outliers = sum(pseudo_labels==-1)
            num_cluster_label = len(set(pl))-1 if num_outliers != 0 else len(set(pl))

            print('==> Create pseudo labels for unlabeled target domain with model_ema')
            target_features_ema = get_features(model_ema,tar_cluster_loader)
            rerank_dist_ema = compute_jaccard_distance(target_features_ema, k1=args.k1, k2=args.k2)
            pseudo_labels_ema = cluster.fit_predict(rerank_dist_ema.astype('double'))
            pl_ema = pseudo_labels_ema.copy()
            num_outliers_ema = sum(pseudo_labels_ema == -1)
            num_cluster_label_ema = len(set(pl_ema)) - 1 if num_outliers_ema != 0 else len(set(pl_ema))

            print('The orignal cluster result: num cluster = {}(model) // {}(model_ema) \t num outliers = {}(model) // '
                  '{}(model_ema)'.format(num_cluster_label,num_cluster_label_ema,num_outliers,num_outliers_ema))
            '''
            reCluster for model and model_ema
            '''
            if args.HC:
                print('Applying hierarchical clustering')
                HC_start_time = time.time()
                max_class = pl.max()
                negative_one_index = (pl == -1).nonzero()
                sc_pseudo_labels = np.delete(pl, negative_one_index)
                dist = np.delete(rerank_dist, negative_one_index, 0)
                dist = np.delete(dist, negative_one_index, 1)
                sc_score_sample = metrics.silhouette_samples(dist, sc_pseudo_labels, metric='precomputed')
                uncer_id = get_uncer_id_by_sc_score(sc_pseudo_labels, sc_score_sample)
                pseudo_labels = reCluster(target_features, pseudo_labels, uncer_id, max_class, args, sc_score_sample)
                num_outliers = sum(pseudo_labels == -1)
                num_cluster_label = len(set(pseudo_labels)) - 1 if num_outliers != 0 else len(set(pseudo_labels))

                max_class_ema = pl_ema.max()
                negative_one_index_ema = (pl_ema == -1).nonzero()
                sc_pseudo_labels_ema = np.delete(pl_ema, negative_one_index_ema)
                dist_ema = np.delete(rerank_dist_ema, negative_one_index_ema, 0)
                dist_ema = np.delete(dist_ema, negative_one_index_ema, 1)
                sc_score_sample_ema = metrics.silhouette_samples(dist_ema, sc_pseudo_labels_ema, metric='precomputed')
                uncer_id_ema = get_uncer_id_by_sc_score(sc_pseudo_labels_ema, sc_score_sample_ema)
                pseudo_labels_ema = reCluster(target_features_ema, pseudo_labels_ema, uncer_id_ema, max_class_ema+50000, args, sc_score_sample_ema)#plus 50000 is for No confusion about pl and pl_ema
                num_outliers_ema = sum(pseudo_labels_ema == -1)
                num_cluster_label_ema = len(set(pseudo_labels_ema)) - 1 if num_outliers_ema != 0 else len(set(pseudo_labels_ema))
                print('HC finish! Cost time={}s'.format(time.time() - HC_start_time))
                print(
                    'The HC Re-cluster result: num cluster = {}(model) // {}(model_ema) \t num outliers = {}(model) // '
                    '{}(model_ema)'.format(num_cluster_label, num_cluster_label_ema, num_outliers, num_outliers_ema))

            '''
            compute uncertain sample index and discard them
            '''
            if args.UCIS:
                print('Applying uncertainty-aware collaborative instance selection')
                UCIS_start_time = time.time()
                pseudo_labels_noNeg1, outlier_oneHot = generate_pseudo_labels(pseudo_labels,num_cluster_label,dataset_target)#number label and one-hot label
                pseudo_labels_noNeg1_ema, outlier_oneHot_ema = generate_pseudo_labels(pseudo_labels_ema, num_cluster_label_ema,dataset_target)
                # ---------------------------------------
                N = pseudo_labels_noNeg1.size(0)

                label_sim = pseudo_labels_noNeg1.expand(N, N).eq(pseudo_labels_noNeg1.expand(N,N).t()).float()  # if label_sim[0]=[1,0,0,1,0],it means sample0 and smaple3 are assigned the same pseudo label
                label_sim_ema = pseudo_labels_noNeg1_ema.expand(N, N).eq(pseudo_labels_noNeg1_ema.expand(N, N).t()).float()  # so label_sim_1[0] may be = [1,0,0,0,1]

                label_sim_new = label_sim - outlier_oneHot
                label_sim_new_ema = label_sim_ema - outlier_oneHot_ema

                label_share = torch.min(label_sim_new,label_sim_new_ema)  # label_sim_new[0]means the first sample's cluster result and its neighbors and so is label_sim_mew_1[0],so we use torch.min
                uncer = label_share.sum(-1) / label_sim.sum(-1)  # model union model_ema / model

                a = torch.le(uncer, 0.8).type(torch.uint8)  # if uncer<0.8,return true
                b = torch.gt(label_sim.sum(-1), 1).type(torch.uint8)  # to check whether when we share the cluster result,any sample degrade to an outlier
                index_zero = a * b  # value 0 means clean
                index_zero = torch.nonzero(index_zero)  # now we get not clean samples' index
                pseudo_labels[index_zero] = -1
                num_noisy_outliers = sum(pseudo_labels == -1)
                num_clean_cluster_label = len(set(pseudo_labels)) - 1 if num_outliers != 0 else len(
                    set(pseudo_labels))
                print('UCIS finish! Cost time={}s'.format(time.time() - UCIS_start_time))
                print(
                    'The UCIS result: num clean cluster = {}(model) \t num outliers = {}(model)'.format(
                        num_clean_cluster_label, num_noisy_outliers))

            cl = list(set(pseudo_labels))
            p1 = []
            new_dataset = []
            cluster_centers = collections.defaultdict(list)
            for i, (item, label) in enumerate(zip(dataset_target.train, pseudo_labels)):
                if label == -1 :
                    continue
                label = cl.index(label)
                p1.append(label)
                new_dataset.append((item[0], label, item[-1]))
                cluster_centers[label].append(target_features[i])
            cluster_features,c_labels = get_cluster_item(cluster_centers)
            cluster_centers.clear()
            model.module.classifier0_32621.weight.data[:len(c_labels)].copy_(
                torch.cat(cluster_features, dim=0).float().cuda())
            model_ema.module.classifier0_32621.weight.data[:len(c_labels)].copy_(
                torch.cat(cluster_features, dim=0).float().cuda())
            ncs = [len(set(p1)) + 1]
            print('mdoel:new class are {}, length of new dataset is {}'.format(ncs, len(new_dataset)))
            del target_features, target_features_ema


        cc = args.choice_c  # (args.choice_c+1)%len(ncs)
        train_loader_target = get_train_loader(dataset_target, args.height, args.width, cc,args.batch_size, args.workers, args.num_instances, iters_, new_dataset)
        # train_loader_source = get_train_loader(dataset_source, args.height, args.width, 0, args.batch_size,args.workers,args.num_instances, args.iters, dataset_source.train)

        # Optimizer
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr , "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)
        # Trainer
        trainer = DbscanBaseTrainer(model, model_ema,num_cluster=ncs, c_name=ncs, alpha=args.alpha, fc_len=fc_len)

        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target,optimizer, args.choice_c,
                      train_iters=iters_, cluster_centers=(cluster_features, c_labels))

        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model' + str(mid) + '_checkpoint.pth.tar'))

        if epoch == 20 or args.dataset_target.find('person') != -1:
            args.eval_step = 2
        elif epoch == 50:
            args.eval_step = 1
        if args.dataset_target.find('market') != -1 or args.dataset_target.find('person') != -1:
            start_eval = 1
        else:
            start_eval = 1

        if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1))  and epoch>start_eval:
            mAP_1 = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                         cmc_flag=False)

            mAP_2 = evaluator_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                             cmc_flag=False)
            is_best = (mAP_1 > best_mAP) or (mAP_2 > best_mAP)
            best_mAP = max(mAP_1, mAP_2, best_mAP)
            save_model(model, (is_best), best_mAP, 1)
            save_model(model_ema, (is_best and (mAP_1 <= mAP_2)), best_mAP, 2)

            print(
                '\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} model no.2(mean-net) mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))

    print('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UCF Training")
    # data
    parser.add_argument('-st', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-tt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--choice_c', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=32621)
    parser.add_argument('--ncs', type=str, default='60')
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--iters', type=int, default=300)

    parser.add_argument('--lambda-value', type=float, default=0)
    # training configs

    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    parser.add_argument('--init-1', type=str,
                        default='logs/market1501TOdukemtmc/resnet50-pretrain-1005/model_best.pth.tar',
                        metavar='PATH')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=8)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir,
                                         'logs/d2m_baseline/resnet50_sbs_gem_memory_ins1005_spbn_sour_debug'))
    parser.add_argument('--log-name',type=str,default='')

    # UCF setting
    parser.add_argument('--HC', action='store_true',
                        help="active the hierarchical clustering (HC) method")
    parser.add_argument('--UCIS', action='store_true',
                        help="active the uncertainty-aware collaborative instance selection (UCIS) method")

    main()
