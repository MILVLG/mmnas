import torch
import torch.nn as nn

class BCE_Loss(nn.Module):
    def __init__(self, __C):
        super(BCE_Loss, self).__init__()
        self.__C = __C
        self.loss_fn_pos = torch.nn.BCELoss(reduction=self.__C.REDUCTION)
        self.loss_fn_negc = torch.nn.BCELoss(reduction=self.__C.REDUCTION)
        self.loss_fn_negi = torch.nn.BCELoss(reduction=self.__C.REDUCTION)

    def forward(self, scores_pos, scores_negc, scores_negi):
        label_pos = torch.ones_like(scores_pos).to(scores_pos.device)
        loss_pos = self.loss_fn_pos(scores_pos, label_pos)

        label_negc = torch.zeros_like(scores_negc).to(scores_negc.device)
        loss_negc = self.loss_fn_negc(scores_negc, label_negc)

        label_negi = torch.zeros_like(scores_negi).to(scores_negi.device)
        loss_negi = self.loss_fn_negi(scores_negi, label_negi)

        loss = loss_pos + loss_negc + loss_pos + loss_negi

        return loss


class Margin_Loss(nn.Module):
    def __init__(self, __C):
        super(Margin_Loss, self).__init__()
        self.__C = __C
        self.margin = 0.2

    def forward(self, scores_pos, scores_negc, scores_negi):
        cost_c = (self.margin + scores_negc - scores_pos).clamp(min=0)
        cost_i = (self.margin + scores_negi - scores_pos).clamp(min=0)

        return cost_c.sum() + cost_i.sum()
