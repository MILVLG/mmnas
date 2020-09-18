import torch, math
import torch.nn as nn
import torch.nn.functional as F
from mmnas.utils.ops_adapter import OpsAdapter
OPS_ADAPTER = OpsAdapter()

class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, pre, s_mask, pre_mask, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_s = s.detach()
        detached_s.requires_grad = True
        with torch.enable_grad():
            output = run_func(detached_s, pre, s_mask, pre_mask)
        ctx.save_for_backward(detached_s, pre, s_mask, pre_mask, output)

        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_s, pre, s_mask, pre_mask, output = ctx.saved_tensors
        if pre is not None:
            grad_s, grad_pre = torch.autograd.grad(output, (detached_s, pre), grad_output, only_inputs=True, allow_unused=True)
        else:
            grad_s, grad_pre = (torch.autograd.grad(output, detached_s, grad_output, only_inputs=True)[0], None)
        binary_grads = ctx.backward_func(detached_s, pre, s_mask, pre_mask, output.data, grad_output.data)

        return grad_s, grad_pre, None, None, binary_grads, None, None


class MixedOp(nn.Module):
    MODE = None

    def __init__(self, __C, name):
        super().__init__()
        self.__C = __C
        if name in OPS_ADAPTER.Used_OPS:
            self.Used_OPS = OPS_ADAPTER.Used_OPS[name]
        else:
            self.Used_OPS = [name]

        self.n_choices = len(self.Used_OPS)

        self.candidate_ops = nn.ModuleList()
        for op_name in self.Used_OPS:
            op = OPS_ADAPTER.OPS[op_name](__C, norm=__C.OPS_NORM, residual=__C.OPS_RESIDUAL)
            self.candidate_ops.append(op)

        self.alpha_prob = nn.Parameter(torch.zeros(self.n_choices))
        self.alpha_gate = nn.Parameter(torch.zeros(self.n_choices))
        self.active_index = None
        self.inactive_index = None

    def forward(self, s, pre=None, s_mask=None, pre_mask=None, rel_embed=None):
        if MixedOp.MODE in ['full', 'two']:
            output = 0
            for _i in self.active_index:
                oi = self.candidate_ops[_i](s, pre, s_mask, pre_mask, rel_embed)
                output = output + self.alpha_gate[_i] * oi

            for _i in self.inactive_index:
                oi = self.candidate_ops[_i](s, pre, s_mask, pre_mask, rel_embed)
                output = output + self.alpha_gate[_i] * oi.detach()

        elif MixedOp.MODE in ['full_v2']:
            assert MixedOp.MODE not in ['full_v2']
            def run_function(candidate_ops, active_id):
                def forward(_s, _pre, _s_mask, _pre_mask):
                    return candidate_ops[active_id](_s, _pre, _s_mask, _pre_mask)

                return forward

            def backward_function(candidate_ops, active_id, binary_gates):
                def backward(_s, _pre, _s_mask, _pre_mask, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                # if _pre is not None:
                                #     out_k = candidate_ops[k](_s.data, _pre.data, _s_mask, _pre_mask)
                                # else:
                                #     out_k = candidate_ops[k](_s.data, _pre, _s_mask, _pre_mask)
                                out_k = candidate_ops[k](_s.data, _pre, _s_mask, _pre_mask)
                            else:
                                out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[k] = grad_k
                    return binary_grads

                return backward

            output = ArchGradientFunction.apply(
                s, pre, s_mask, pre_mask, self.alpha_gate,
                run_function(self.candidate_ops, self.active_index[0]),
                backward_function(self.candidate_ops, self.active_index[0], self.alpha_gate)
            )

        else:
            output = self.active_op(s, pre, s_mask, pre_mask, rel_embed)

        return output


    @property
    def probs_over_ops(self):
        probs = F.softmax(self.alpha_prob, dim=0)  # softmax to probability
        return probs

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def binarize(self):
        # reset binary gates
        self.alpha_gate.data.zero_()
        probs = self.probs_over_ops

        if MixedOp.MODE in ['two']:
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([self.alpha_prob[idx] for idx in sample_op]), dim=0)

            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.alpha_gate.data[active_op] = 1.0

        else:
            sample = torch.multinomial(probs.data, 1)[0].item()
            # ix_v, ix = torch.topk(probs.data, 1)
            # print(probs)
            # print(sample, ix[0].item(), ix_v[0].item())
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + [_i for _i in range(sample + 1, self.n_choices)]
            # set binary gate
            self.alpha_gate.data[sample] = 1.0

        # avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def delta_ij(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def set_arch_param_grad(self):
        binary_grads = self.alpha_gate.grad.data
        # if self.active_op.is_zero_layer():
        #     self.alpha_prob.grad = None
        #     return
        if self.alpha_prob.grad is None:
            self.alpha_prob.grad = torch.zeros_like(self.alpha_prob.data)

        if MixedOp.MODE in ['two']:
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([self.alpha_prob[idx] for idx in involved_idx]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.alpha_prob.grad.data[origin_i] += binary_grads[origin_j] * probs_slice[j] * (self.delta_ij(i, j) - probs_slice[i])

            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx, self.alpha_prob.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx, self.alpha_prob.data[idx].item())

        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.alpha_prob.grad.data[i] += binary_grads[j] * probs[j] * (self.delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.alpha_prob.data[idx] for idx in involved_idx]
        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )
        for idx in involved_idx:
            self.alpha_prob.data[idx] -= offset
