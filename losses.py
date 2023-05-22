"""
Adapted from: https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        #print(len(features), features.shape, len(labels))
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        #print("features.is_cuda", features.is_cuda)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            #print(labels.shape)
            labels = labels.contiguous().view(-1, 1)
            #print(labels.shape)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #lebels = torch.unsqueeze(labels, -1)
            #print("here", labels.shape)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #print("t", contrast_feature.shape, mask.shape, labels)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] #take only the first btz features
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature #2*btz
            anchor_count = contrast_count #2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        #print("mask1", mask.shape)
        mask = mask.repeat(anchor_count, contrast_count)
        #print("mask2", mask.shape)
        # mask-out self-contrast cases
        #print("mask3", torch.ones_like(mask), torch.arange(batch_size * anchor_count).view(-1, 1))
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        #print("mask4", labels.shape, torch.ones_like(mask), torch.arange(batch_size * anchor_count).view(-1, 1))
        mask = mask * logits_mask

        #print("hhh", mask.sum(1))
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #print("loss11", mean_log_prob_pos)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        #print("loss1", loss)
        loss = loss.view(anchor_count, batch_size).mean()
        #print("loss2", loss)
        return loss
