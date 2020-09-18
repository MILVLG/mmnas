import torch, math
import torch.nn as nn
import torch.nn.functional as F

''' 
==================
    Operations
==================
'''

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.linear_merge = nn.Linear(__C.HSIZE * __C.ATTFLAT_GLIMPSES, __C.ATTFLAT_OUT_SIZE)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return x


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return x * 0.


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GatedLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatedLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return self.glu(self.linear(x))


class GLU(nn.Module):
    def __init__(self, __C, norm=False, residual=False, layers=1):
        super(GLU, self).__init__()
        assert layers in [1, 2]
        self.layers = layers
        self.norm = norm
        self.residual = residual

        if layers == 1:
            self.unit = GatedLinear(__C.HSIZE, __C.HSIZE)
        else:
            self.unit_0 = GatedLinear(__C.HSIZE, __C.HSIZE * 2)
            self.unit_1 = GatedLinear(__C.HSIZE * 2, __C.HSIZE)
            self.dropout_u = nn.Dropout(__C.DROPOUT_R)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        if self.layers == 1:
            x_att = self.dropout(self.unit(x))
        else:
            x_att = self.dropout(self.unit_1(self.dropout_u(F.relu(self.unit_0(x)))))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class MHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class RelMHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(RelMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_r = nn.Linear(__C.REL_SIZE, self.HHEAD, bias=True)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, k, q, mask=None, rel_embed=None):
        assert rel_embed is not None
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        r = self.relu(self.linear_r(rel_embed)).permute(0, 3, 1, 2)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.log(torch.clamp(r, min=1e-6)) + scores
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        atted = torch.matmul(att_map, v)

        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted


class SelfAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(SelfAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.mhatt(x, x, x, x_mask))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class RelSelfAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(RelSelfAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = RelMHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        assert rel_embed is not None
        x_att = self.dropout(self.mhatt(x, x, x, x_mask, rel_embed))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class GuidedAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(GuidedAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        assert y is not None
        x_att = self.dropout(self.mhatt(y, y, x, y_mask))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, __C, norm=False, residual=False, mid_k=None):
        super(FeedForward, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        if mid_k:
            self.MID_SIZE = __C.HSIZE * mid_k
        else:
            self.MID_SIZE = __C.HSIZE * 4

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=self.MID_SIZE,
            out_size=__C.HSIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.mlp(x))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class FeedForward_deep(nn.Module):
    def __init__(self, __C, norm=False, residual=False, mid_k=None):
        super(FeedForward_deep, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        if mid_k:
            self.MID_SIZE = __C.HSIZE * mid_k
        else:
            self.MID_SIZE = __C.HSIZE * 2

        self.fc = FC(__C.HSIZE, self.MID_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True)
        self.mlp = MLP(
            in_size=self.MID_SIZE,
            mid_size=self.MID_SIZE,
            out_size=__C.HSIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.mlp(self.fc(x)))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class UniimgAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(UniimgAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        assert y is not None
        xy = torch.cat((x, y), dim=1)
        x_att = self.dropout(self.mhatt(xy, xy, x))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class SepConv(nn.Module):
    def __init__(self, __C, norm=False, residual=False, k=3):
        super(SepConv, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.depthwise_conv = nn.Conv1d(in_channels=__C.HSIZE, out_channels=__C.HSIZE, kernel_size=k, groups=__C.HSIZE,
                                        padding=k // 2, bias=True)
        self.pointwise_conv = nn.Conv1d(in_channels=__C.HSIZE, out_channels=__C.HSIZE, kernel_size=1, padding=0, bias=True)

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.pointwise_conv(self.depthwise_conv(x.transpose(1, 2))).transpose(1, 2))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class StdConv(nn.Module):
    def __init__(self, __C, norm=False, residual=False, k=3):
        super(StdConv, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.conv = nn.Conv1d(in_channels=__C.HSIZE, out_channels=__C.HSIZE, kernel_size=k, padding=k // 2, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.conv(x.transpose(1, 2)).transpose(1, 2))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x
