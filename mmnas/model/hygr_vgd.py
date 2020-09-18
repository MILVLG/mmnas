from mmnas.utils.ops_adapter import OpsAdapter
from mmnas.model.mixed import MixedOp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmnas.model.modules import AttFlat, LayerNorm

OPS_ADAPTER = OpsAdapter()

class Cell_Search(nn.Module):
    def __init__(self, __C, type):
        super(Cell_Search, self).__init__()
        self.__C = __C

        self.dag = nn.ModuleList()
        for i in range(__C.NODES[type]):
            self.dag.append(nn.ModuleList())
            op1 = MixedOp(__C, type + '_safe')
            self.dag[i].append(op1)

    def forward(self, s, pre=None, s_mask=None, pre_mask=None, rel_embed=None):
        for ops in self.dag:
            s = sum(op(s, pre, s_mask, pre_mask, rel_embed) for op in ops)

        return s


class Backbone_Search(nn.Module):
    def __init__(self, __C):
        super(Backbone_Search, self).__init__()
        self.__C = __C

        self.cells_enc = nn.ModuleList()
        for i in range(__C.LAYERS):
            cell_enc = Cell_Search(__C, type='enc')
            self.cells_enc.append(cell_enc)

        self.cells_dec = nn.ModuleList()
        for i in range(__C.LAYERS):
            cell_dec = Cell_Search(__C, type='dec')
            self.cells_dec.append(cell_dec)

    def forward(self, x, y, x_mask, y_mask, x_rel_embed, y_rel_embed):
        for i, cell in enumerate(self.cells_enc):
            x = cell(s=x, s_mask=x_mask, rel_embed=x_rel_embed)

        for i, cell in enumerate(self.cells_dec):
            y = cell(s=y, pre=x, s_mask=y_mask, pre_mask=x_mask, rel_embed=y_rel_embed)

        return x, y


class Net_Search(nn.Module):
    def __init__(self, __C, init_dict):
        super(Net_Search, self).__init__()
        self.__C = __C
        self._redundant_modules = None
        self._unused_modules = None

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

        self.backnone = Backbone_Search(__C)
        self.attflat_x = AttFlat(__C)
        self.attfc_y = nn.Linear(__C.HSIZE, __C.ATTFLAT_OUT_SIZE)
        self.proj_norm = LayerNorm(__C.ATTFLAT_OUT_SIZE)
        self.proj_scores = nn.Linear(__C.ATTFLAT_OUT_SIZE, 1)
        self.proj_reg = nn.Linear(__C.ATTFLAT_OUT_SIZE, 4)
        self.linear_y_rel = nn.Linear(4, __C.REL_SIZE)

        self.init_arch()
        self._net_weights = []
        for n, p in self.named_parameters():
            if 'alpha_prob' not in n and 'alpha_gate' not in n:
                self._net_weights.append((n, p))

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
        x_out = self.attflat_x(x_out, x_mask).unsqueeze(1)
        y_out = self.attfc_y(y_out)

        xy_out = x_out + y_out
        xy_out = self.proj_norm(xy_out)
        pred_scores = self.proj_scores(xy_out).squeeze(-1)
        if self.__C.SCORES_LOSS == 'kld':
            pred_scores = F.log_softmax(pred_scores, dim=-1)
        pred_reg = self.proj_reg(xy_out)

        return pred_scores, pred_reg

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)

    def init_arch(self):
        # setup alphas list
        self._alphas_prob = []
        for n, p in self.named_parameters():
            if 'alpha_prob' in n:
                self._alphas_prob.append((n, p))

        self._alphas_gate = []
        for n, p in self.named_parameters():
            if 'alpha_gate' in n:
                self._alphas_gate.append((n, p))

        for param in self.alpha_prob_parameters():
            if self.__C.ALPHA_INIT_TYPE == 'normal':
                param.data.normal_(0, 1e-3)
            elif self.__C.ALPHA_INIT_TYPE == 'uniform':
                param.data.uniform_(-1e-3, 1e-3)

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedOp'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedOp.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def alpha_prob_parameters(self):
        for n, p in self._alphas_prob:
            yield p

    def alpha_gate_parameters(self):
        for n, p in self._alphas_gate:
            yield p

    def named_alpha_prob_parameters(self):
        for n, p in self._alphas_prob:
            yield n, p

    def named_alpha_gate_parameters(self):
        for n, p in self._alphas_gate:
            yield n, p

    def net_parameters(self):
        for n, p in self._net_weights:
            yield p

    def named_net_parameters(self):
        for n, p in self._net_weights:
            yield n, p

    def genotype(self):
        alpha_enc = []
        alpha_dec = []
        for n, p in self.named_alpha_prob_parameters():
            if 'enc' in n:
                alpha_enc.append(p)
            if 'dec' in n:
                alpha_dec.append(p)

            gene_enc = self.parse(alpha_enc, type='enc')
            gene_dec = self.parse(alpha_dec, type='dec')

        return {
            'enc': gene_enc,
            'dec': gene_dec,
        }

    def parse(self, alpha_param_list, type):
        gene = []
        # assert OPS_ADAPTER.Used_OPS[type][-1] == 'none'  # assume last PRIMITIVE is 'none'

        node_gene = []
        for ix, edges in enumerate(alpha_param_list):
            # edges: Tensor(n_edges, n_ops)
            edge_max, op_indices = torch.topk(edges, 1)  # choose top-1 op in every edge
            op_name = OPS_ADAPTER.Used_OPS[type][op_indices[0]]
            node_gene.append(op_name)

            gene.append(node_gene)
            node_gene = []

        return gene


    def genotype_weights(self):
        alpha_enc = []
        alpha_dec = []
        for n, p in self.named_alpha_prob_parameters():
            if 'enc' in n:
                alpha_enc.append(p)
            if 'dec' in n:
                alpha_dec.append(p)

        weights_enc = self.parse_weights(alpha_enc)
        weights_dec = self.parse_weights(alpha_dec)

        return {
            'w_enc': weights_enc,
            'w_dec': weights_dec,
        }

    def parse_weights(self, alpha):
        with torch.no_grad():
            weights = [F.softmax(a, dim=-1).data.cpu().numpy() for a in alpha]

        return weights
