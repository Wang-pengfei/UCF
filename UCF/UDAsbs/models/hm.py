import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            if y >= 0:
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))



# def nll_loss(inputs, targets, log_input=True, full=False, size_average=None, eps=1e-8,
#                      reduce=None, reduction='mean'):
#     log_probs = inputs
#     # targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
#     targets = targets.cuda()
#     loss = (- targets * log_probs).mean(0).sum()
#     # print(loss)
#     return loss

def nll_loss(inputs, targets, weight, log_input=True, full=False, size_average=None, eps=1e-8,
                     reduce=None, reduction='mean'):
    log_probs = inputs
    # targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = targets.cuda()
    loss = (- targets * log_probs)*(weight.unsqueeze(1))
    # print(loss)
    loss = (loss.mean(0)).sum()

    return loss



class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        # self.targets = targets

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('labels_1', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, targets, weight):
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp
        # print(inputs.size())
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        # targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        labels_1 = self.labels_1.clone()
        hard_index = torch.le(labels_1, 0).type(torch.uint8).cuda()
        # print(hard_index)

        input_sim = inputs*hard_index
        input_num = torch.ones(self.num_samples, 1) * hard_index.unsqueeze(1).cpu()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, input_sim.t().contiguous())

        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, input_num.float().cuda())
        mask = (nums>0).float()

        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)   #average

        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())

        # print(targets)

        # return F.nll_loss(torch.log(masked_sim+1e-6), targets)
        return nll_loss(torch.log(masked_sim+1e-6), targets, weight)
