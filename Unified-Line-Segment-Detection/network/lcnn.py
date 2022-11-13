import torch
import torch.nn as nn
import torch.nn.functional as F
from network.stacked_hourglass_network import StackedHourglassNetwork
from network.line_proposal_network import LineProposalNetwork
from network.loi_head import LoIHead


class LCNN(nn.Module):
    def __init__(self, cfg):
        super(LCNN, self).__init__()

        output_channels_list = [1, 1, 2, 1, 2,] + [4] * ((cfg.order + 1) // 2)

        self.extractor = StackedHourglassNetwork(
            depth=cfg.depth,
            num_stacks=cfg.num_stacks,
            num_blocks=cfg.num_blocks
        )
        self.lpn = LineProposalNetwork(
            input_channels=cfg.num_feats,
            output_channels_list=output_channels_list,
            order=cfg.order,
            junc_thresh=cfg.junc_thresh,
            junc_max_num=cfg.junc_max_num,
            line_max_num=cfg.line_max_num,
            num_pos_proposals=cfg.num_pos_proposals,
            num_neg_proposals=cfg.num_neg_proposals,
            nms_size=cfg.nms_size
        )
        self.head = LoIHead(num_feats=cfg.num_feats, order=cfg.order, n_pts=cfg.n_pts)

    def postprocess(self, loi_preds, loi_scores, thresh=0.0):
        line_preds, line_scores = [], []
        for loi_pred, loi_score in zip(loi_preds, loi_scores):
            keep_id = loi_score >= thresh
            line_pred = loi_pred[keep_id]
            line_score = loi_score[keep_id]
            line_preds.append(line_pred)
            line_scores.append(line_score)

        return line_preds, line_scores

    def forward(self, images, metas=None):
        features = self.extractor(images)
        maps, loi_preds = self.lpn(features, metas)

        if self.training:
            loi_preds, loi_labels = self.lpn.sample_lines(loi_preds, metas)
            loi_scores = self.head(features, loi_preds)
            return maps, loi_scores, loi_labels
        else:
            loi_scores = self.head(features, loi_preds)
            line_preds, line_scores = self.postprocess(loi_preds, loi_scores)
            return maps['jmap'], maps['joff'], line_preds, line_scores


def weighted_l1_loss(logits, target, mask=None):
    loss = F.l1_loss(logits, target, reduction='none')
    if mask is not None:
        w = mask.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean()


def weighted_smooth_l1_loss(logits, target, mask=None):
    loss = F.smooth_l1_loss(logits, target, reduction='none')
    if mask is not None:
        loss = loss * mask
    return loss.mean()


def lpn_loss_func(outputs, labels):
    """
    LPN Loss

    """
    lmap_loss = F.binary_cross_entropy(outputs['lmap'], labels['lmap'])
    jmap_loss = F.binary_cross_entropy(outputs['jmap'], labels['jmap'])
    joff_loss = weighted_l1_loss(outputs['joff'], labels['joff'], labels['jmap'])
    cmap_loss = F.binary_cross_entropy(outputs['cmap'], labels['cmap'])
    coff_loss = weighted_l1_loss(outputs['coff'], labels['coff'], labels['cmap'])

    lvec_loss = 0.0
    for i in range(outputs['lvec'].shape[1] // 2):
        lvec_loss += weighted_smooth_l1_loss(outputs['lvec'][:, 2 * i:2 * (i + 1)],
             labels['lvec'][:, 2 * i:2 * (i + 1)], labels['cmap'])

    return lmap_loss, jmap_loss, joff_loss, cmap_loss, coff_loss, lvec_loss


def loi_loss_func(outputs, labels):
    """
    LoI Head Loss

    """
    loi_loss, pos_loss, neg_loss = 0.0, 0.0, 0.0
    batch_size = len(outputs)
    for output, label in zip(outputs, labels):
        loss = F.binary_cross_entropy(output, label, reduction='none')
        pos_loss += loss[label == 1].mean() / batch_size
        neg_loss += loss[label == 0].mean() / batch_size

    loi_loss = 1.0 * pos_loss + 1.0 * neg_loss

    return loi_loss
