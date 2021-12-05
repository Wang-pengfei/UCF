from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from ..layers import (
    IBN,
    Non_local,
    get_norm,
)

from .gem_pooling import GeneralizedMeanPoolingP

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(2048, 1024)
        # self.fc1_2 = nn.Linear(2048, 1024)
        # self.fc2 = nn.Linear(2048, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 6)

    # forward method
    def forward(self, input):
        x = F.relu(self.fc1_bn(self.fc1_1(input.view(input.size(0), -1))))
        # y = F.leaky_relu(self.fc1_2(input.view(input.size(0), -1)), 0.2)

        # y = F.leaky_relu(self.fc1_2(label), 0.2)
        # x = torch.cat([x, y], 1)
        # x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, mb_h=2048, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool)  # no relu
        # with_nl = True
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gap = GeneralizedMeanPoolingP(3)
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes is not None:
                for i, num_cluster in enumerate(self.num_classes):
                    exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i, num_cluster,
                                                                                                      num_cluster))
                    exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i, num_cluster))

                # for i, j in enumerate([702]):
                #     exec("self.classifier_s{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i, 702,
                #                                                                                       702))
                #     exec("init.normal_(self.classifier_s{}_{}.weight, std=0.001)".format(i, 702))

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False, training=False, cluster=False, source=False):
        x = self.base(x)
        x4 = self.layer1(x)
        x3 = self.layer2(x4)
        x2 = self.layer3(x3)
        x1 = self.layer4(x2)

        x = self.gap(x1)
        x = x.view(x.size(0), -1)
        if self.cut_at_pooling: return x  # FALSE
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))  # FALSE
        else:
            bn_x = self.feat_bn(x)  # 1

        if training is False:
            bn_x = self.feat_bn(x)
            bn_x = F.normalize(bn_x)
            return bn_x
        if self.norm:  # FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:  # FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:  # FALSE
            bn_x = self.drop(bn_x)

        prob = []

        if source is False:
            for i, num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i, num_cluster))


        if feature_withbn:  # False
            return bn_x, prob

        return x, prob, bn_x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(mb_h, **kwargs):
    return ResNet(50, mb_h=mb_h, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
