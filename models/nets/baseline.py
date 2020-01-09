import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, net_enc, net_dec, deep_sup_scale=None):
        super(Net, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.deep_sup_scale = deep_sup_scale
        self.ohem_hard_thres = 0.7
        self.ohem_min_pixels = 0.2

    def forward(self, feed_dict):
        ret_dict = self.decoder(self.encoder(feed_dict['img_data']))

        '''
        Update start from here
        '''
        logit = ret_dict['logit']
        logit = nn.functional.interpolate(logit, scale_factor=4, mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logit, feed_dict['seg_label'], ignore_index=-1, reduction='none')

        # online hard examples mining
        with torch.no_grad():
            # online hard examples mining
            B, H, W = feed_dict['seg_label'].shape
            bidx, hidx, widx = torch.meshgrid(torch.arange(B), torch.arange(H), torch.arange(W))
            prob = F.softmax(logit, dim=1)
            gt_prob = torch.exp(prob[bidx, feed_dict['seg_label'], hidx, widx])
            hard_mask = (gt_prob < self.ohem_hard_thres) & (feed_dict['seg_label'] >= 0)
            min_pixels = int(self.ohem_min_pixels * (feed_dict['seg_label'] >= 0).sum().item())
        if hard_mask.sum().item() > min_pixels:
            loss = loss[hard_mask].mean()
        else:
            loss = loss.reshape(-1).topk(min_pixels)[0].mean()

        return loss, logit

