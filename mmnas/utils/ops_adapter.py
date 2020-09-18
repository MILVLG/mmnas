import torch.nn as nn
from mmnas.model.modules import *


class OpsAdapter:
    def __init__(self):
        self.Used_OPS = {
            'enc_safe': [
                # 'skip_connect',  # identity
                'self_att_64',
                'feed_forward',
            ],
            'dec_safe': [
                # 'skip_connect',  # identity
                'self_att_64',
                'rel_self_att_64',
                'guided_att_64',
                'feed_forward',
            ],
        }
        self.Used_OPS['enc'] = self.Used_OPS['enc_safe'] + ['none']
        self.Used_OPS['dec'] = self.Used_OPS['dec_safe'] + ['none']

        self.OPS = {
            'none': lambda __C, norm, residual: Zero(),
            'skip_connect': lambda __C, norm, residual: Identity(),
            'relu': lambda __C, norm, residual: nn.ReLU(),
            'gelu': lambda __C, norm, residual: GELU(),
            'leakyrelu': lambda __C, norm, residual: nn.LeakyReLU(),

            'self_att_256': lambda __C, norm, residual: SelfAtt(__C, norm, residual, base=256),
            'self_att_128': lambda __C, norm, residual: SelfAtt(__C, norm, residual, base=128),
            'self_att_64': lambda __C, norm, residual: SelfAtt(__C, norm, residual, base=64),
            'self_att_32': lambda __C, norm, residual: SelfAtt(__C, norm, residual, base=32),
            'self_att_16': lambda __C, norm, residual: SelfAtt(__C, norm, residual, base=16),
            'self_att_64_2': lambda __C, norm, residual: SelfAtt(__C, norm, residual, base=64, hsize_k=2),

            'rel_self_att_256': lambda __C, norm, residual: RelSelfAtt(__C, norm, residual, base=256),
            'rel_self_att_128': lambda __C, norm, residual: RelSelfAtt(__C, norm, residual, base=128),
            'rel_self_att_64': lambda __C, norm, residual: RelSelfAtt(__C, norm, residual, base=64),
            'rel_self_att_32': lambda __C, norm, residual: RelSelfAtt(__C, norm, residual, base=32),
            'rel_self_att_16': lambda __C, norm, residual: RelSelfAtt(__C, norm, residual, base=16),

            'guided_att_256': lambda __C, norm, residual: GuidedAtt(__C, norm, residual, base=256),
            'guided_att_128': lambda __C, norm, residual: GuidedAtt(__C, norm, residual, base=128),
            'guided_att_64': lambda __C, norm, residual: GuidedAtt(__C, norm, residual, base=64),
            'guided_att_32': lambda __C, norm, residual: GuidedAtt(__C, norm, residual, base=32),
            'guided_att_16': lambda __C, norm, residual: GuidedAtt(__C, norm, residual, base=16),
            'guided_att_64_2': lambda __C, norm, residual: GuidedAtt(__C, norm, residual, base=64, hsize_k=2),

            'uniimg_att_128': lambda __C, norm, residual: UniimgAtt(__C, norm, residual, base=128),
            'uniimg_att_64': lambda __C, norm, residual: UniimgAtt(__C, norm, residual, base=64),
            'uniimg_att_32': lambda __C, norm, residual: UniimgAtt(__C, norm, residual, base=32),

            'sep_conv_3': lambda __C, norm, residual: SepConv(__C, norm, residual, k=3),
            'sep_conv_5': lambda __C, norm, residual: SepConv(__C, norm, residual, k=5),
            'sep_conv_7': lambda __C, norm, residual: SepConv(__C, norm, residual, k=7),
            'sep_conv_11': lambda __C, norm, residual: SepConv(__C, norm, residual, k=11),

            'std_conv_3': lambda __C, norm, residual: StdConv(__C, norm, residual, k=3),
            'std_conv_5': lambda __C, norm, residual: StdConv(__C, norm, residual, k=5),
            'std_conv_7': lambda __C, norm, residual: StdConv(__C, norm, residual, k=7),
            'std_conv_11': lambda __C, norm, residual: StdConv(__C, norm, residual, k=11),

            'feed_forward_2': lambda __C, norm, residual: FeedForward(__C, norm, residual, mid_k=2),
            'feed_forward': lambda __C, norm, residual: FeedForward(__C, norm, residual),
            'feed_forward_8': lambda __C, norm, residual: FeedForward(__C, norm, residual, mid_k=8),
            'feed_forward_16': lambda __C, norm, residual: FeedForward(__C, norm, residual, mid_k=16),
            'feed_forward_32': lambda __C, norm, residual: FeedForward(__C, norm, residual, mid_k=32),
            'gated_linear_1': lambda __C, norm, residual: GLU(__C, norm, residual, layers=1),
            'gated_linear_2': lambda __C, norm, residual: GLU(__C, norm, residual, layers=2),
            'feed_forward_deep': lambda __C, norm, residual: FeedForward_deep(__C, norm, residual),

        }