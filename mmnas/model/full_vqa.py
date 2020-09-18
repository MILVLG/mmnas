from mmnas.utils.ops_adapter import OpsAdapter
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmnas.model.modules import AttFlat, LayerNorm

OPS_ADAPTER = OpsAdapter()

class Cell_Full(nn.Module):
    def __init__(self, __C, type):
        super(Cell_Full, self).__init__()
        self.__C = __C
        self.NODES = len(__C.GENOTYPE[type])

        gene = __C.GENOTYPE[type]
        self.dag = nn.ModuleList()
        for node in gene:
            row = nn.ModuleList()
            for op_name in node:
                op = OPS_ADAPTER.OPS[op_name](__C, norm=__C.OPS_NORM, residual=__C.OPS_RESIDUAL)
                row.append(op)
            self.dag.append(row)

    def forward(self, s, pre=None, s_mask=None, pre_mask=None, rel_embed=None):
        for ops in self.dag:
            s = sum(op(s, pre, s_mask, pre_mask, rel_embed) for op in ops)

        return s


class Backbone_Full(nn.Module):
    def __init__(self, __C):
        super(Backbone_Full, self).__init__()
        self.__C = __C

        self.cells_enc = nn.ModuleList()
        for i in range(__C.LAYERS):
            cell_enc = Cell_Full(__C, type='enc')
            self.cells_enc.append(cell_enc)

        self.cells_dec = nn.ModuleList()
        for i in range(__C.LAYERS):
            cell_dec = Cell_Full(__C, type='dec')
            self.cells_dec.append(cell_dec)

    def forward(self, x, y, x_mask, y_mask, x_rel_embed, y_rel_embed):
        for i, cell in enumerate(self.cells_enc):
            x = cell(s=x, s_mask=x_mask, rel_embed=x_rel_embed)

        for i, cell in enumerate(self.cells_dec):
            y = cell(s=y, pre=x, s_mask=y_mask, pre_mask=x_mask, rel_embed=y_rel_embed)

        return x, y


class Net_Full(nn.Module):
    def __init__(self, __C, init_dict):
        super(Net_Full, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(num_embeddings=init_dict['token_size'], embedding_dim=__C.WORD_EMBED_SIZE)
        self.embedding.weight.data.copy_(torch.from_numpy(init_dict['pretrained_emb']))
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HSIZE,
            num_layers=1,
            batch_first=True
        )

        imgfeat_linear_size = __C.FRCNFEAT_SIZE
        if __C.BBOX_FEATURE:
            self.bboxfeat_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.imgfeat_linear = nn.Linear(imgfeat_linear_size, __C.HSIZE)

        self.backnone = Backbone_Full(__C)
        self.attflat_x = AttFlat(__C)
        self.attflat_y = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.ATTFLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.ATTFLAT_OUT_SIZE, init_dict['ans_size'])

        self.linear_y_rel = nn.Linear(4, __C.REL_SIZE)


    def forward(self, input):
        frcn_feat, bbox_feat, y_rel_embed, ques_ix, x_rel_embed = input

        # with torch.no_grad():
        # Make mask for attention learning
        x_mask = self.make_mask(ques_ix.unsqueeze(2))
        y_mask = self.make_mask(frcn_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        x_in, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        if self.__C.BBOX_FEATURE:
            bbox_feat = self.bboxfeat_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        y_in = self.imgfeat_linear(frcn_feat)

        y_rel_embed = F.relu(self.linear_y_rel(y_rel_embed))
        x_out, y_out = self.backnone(x_in, y_in, x_mask, y_mask, x_rel_embed, y_rel_embed)
        x_out = self.attflat_x(x_out, x_mask)
        y_out = self.attflat_y(y_out, y_mask)
        xy_out = x_out + y_out
        xy_out = self.proj_norm(xy_out)
        proj_feat = self.proj(xy_out)

        return proj_feat

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)

