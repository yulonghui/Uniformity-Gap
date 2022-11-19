import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import time
import random

class HUG_MHE(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super(HUG_MHE, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, feat, classifier, target):
        feat = feat
        weight = classifier
        weight = weight / torch.norm(weight, 2, 1, keepdim=True)
        inner_pro = torch.cdist(weight, weight)
        inner_pro = torch.triu(inner_pro, diagonal=1)
        pro_mask = inner_pro > 0 
        weight_wise = torch.mean(1. / (inner_pro[pro_mask] * inner_pro[pro_mask]))
        sample_wise_all = 0
        class_indices = torch.unique(target)
        for class_i in class_indices:
            mask = target == class_i
            feat_class = feat[mask]
            feat_class = feat_class / torch.norm(feat_class, 2, 1, keepdim=True)
            class_mean = weight[class_i].unsqueeze(0)
            concen_loss = torch.cdist(feat_class, class_mean.detach())
            sample_wise = torch.mean(concen_loss)
            sample_wise_all += sample_wise
        return sample_wise_all/len(class_indices) * self.beta, weight_wise * self.alpha, feat

if __name__ == '__main__':
    # utilizing the random data to simulate the unconstrained feature
    # simulate CIFAR-100
    num_c = 100
    sample_c = 500 

    all_features = Parameter(torch.Tensor(50000, 512).cuda())
    stdv = 1. / math.sqrt(all_features.size(1))
    all_features.data.uniform_(-stdv, stdv)
    all_features.requires_grad = True

    all_classifier = Parameter(torch.Tensor(num_c, 512).cuda())
    stdv = 1. / math.sqrt(all_classifier.size(1))
    all_classifier.data.uniform_(-stdv, stdv)
    all_classifier.requires_grad = True

    all_labels = None

    optimizer = torch.optim.SGD([{"params":all_features}, {"params":all_classifier}], 1.0,
                                momentum=0.9, nesterov = True,
                                weight_decay=5e-4)
    for class_i in range(num_c):
        class_tensor = class_i * torch.ones(sample_c, 1)
        if all_labels == None:
            all_labels = class_tensor
        else:
            all_labels = torch.cat((all_labels, class_tensor), 0)
    criterion = HUG_MHE()
    for i in range(200):
        for j in range(100):
            feat = all_features.cuda()
            labels = all_labels.squeeze().long().cuda()
            loss1, loss2, _ = decouple_loss_mhs(feat, all_classifier, labels)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss1.item(), loss2.item())
