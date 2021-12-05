from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, SoftTripletLoss, CrossEntropyLabelSmooth_s, \
    CrossEntropyLabelSmooth_c, SoftEntropy
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.criterion_ce_s = CrossEntropyLabelSmooth_s(num_classes).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)

            s_features, s_cls_out, _ = self.model(s_inputs, training=True)
            # target samples: only forward
            _, _, _ = self.model(t_inputs, training=True)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out[0], targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs  # , pids, index
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce_s(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class DbscanBaseTrainer(object):
    def __init__(self, model, model_ema, num_cluster=None, c_name=None, alpha=0.999, fc_len=3000):
        super(DbscanBaseTrainer, self).__init__()
        self.model= model
        self.num_cluster = num_cluster
        self.c_name = [fc_len for _ in range(len(num_cluster))]
        self.model_ema = model_ema
        self.alpha = alpha
        self.criterion_ce = nn.CrossEntropyLoss().cuda()
        self.criterion_tri = TripletLoss(margin=0.3).cuda()
        self.softmax = nn.Softmax(dim=-1)

    def train(self, epoch, data_loader_target, optimizer,  choice_c,
               train_iters=200, cluster_centers=None):

        self.model.train()
        self.model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce_clean = AverageMeter()
        losses_ct_clean = AverageMeter()

        precisions = [AverageMeter(), AverageMeter()]
        cluster_features, cluster_labels = cluster_centers
        num_clusters = len(cluster_labels)

        cf = torch.cat(cluster_features, dim=0).cuda()
        cl = torch.Tensor(cluster_labels).cuda()

        memory = HybridMemory(2048, num_clusters,
                              0.05, momentum=0.2).cuda()
        memory.features = cf
        memory.labels = cl

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            # process inputs
            imgs,targets = self._parse_data(target_inputs)
            f_out_t1, p_out_t1, memory_f_t1 = self.model(imgs, training=True, source=False)
            p_out_t1[0] = p_out_t1[0][:, :num_clusters]
            with torch.no_grad():
                f_out_t1_ema, p_out_t1_ema, memory_f_t1_ema= self.model_ema(imgs, training=True)
                p_out_t1_ema[0] = p_out_t1_ema[0][:, :num_clusters]
            with torch.no_grad():
                cf = memory.features.clone().detach()
            memory_f_t1 = F.normalize(memory_f_t1, dim=1)
            sim_f = torch.mm(memory_f_t1, cf.t()) / 0.05
            loss_ct = F.cross_entropy(sim_f, targets)

            loss_ce = self.criterion_ce(p_out_t1[0],targets)
            loss = loss_ce+loss_ct

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model, self.model_ema, self.alpha, epoch * len(data_loader_target) + i)
            '''
            memory update
            '''
            memory.update(memory_f_t1, targets)
            prec_1, = accuracy(p_out_t1[0].data, targets.data)

            losses_ce_clean.update(loss_ce.item())
            losses_ct_clean.update(loss_ct.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            # 'Loss_camera {:.3f} \t'
            if (i + 1) % 50 == 1:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce_clean {:.3f} \t'         
                      'Loss_ct_clean {:.3f} \t'        
                      'Prec {:.2%}'
                      .format(epoch, i, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce_clean.avg,
                              losses_ct_clean.avg,
                              precisions[0].avg))

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for shufflebn"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



    def _parse_data(self, inputs):
        # [img, fname]+ pids+[camid, inds]
        imgs = inputs[0]
        pids = inputs[2]
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets



class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def farward(self):
        return 0

    @torch.no_grad()
    def update(self, f_out, p_labels):
        for x, y in zip(f_out, p_labels):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()
