import torch
import torch.nn as nn
from block import FC_block

sp = torch.zeros((1, 1))
counter = 0

class SNN(nn.Module):
    def __init__(self, hyperparams, hidden_size, layers=3, sbp=False, bp_mark=None):
        super(SNN, self).__init__()
        self.hyperparams = hyperparams
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layers_size = [self.hyperparams[1]] + [self.hidden_size] * (layers - 2)  + [self.hyperparams[2], None]
        print('layers size:', self.layers_size[:-1])
        self.len = len(self.layers_size) - 2
        self.error = None

        for i in range(self.len):
            self.layers.append(FC_block(self.hyperparams, self.layers_size[i], self.layers_size[i + 1], self.layers_size[i + 2]))

            def hook_fc_backward(module, grad_input, grad_output):
                global sp, counter
                # grad_input:[grad_b, grad_x, grad_w]
                grad_b = grad_input[0]
                grad_w = grad_input[2]

                if module.last_layer:
                    # calculate inductor 'sp' from gradients of the last layer
                    counter += 1
                    sp = torch.diag(torch.sum(grad_w.abs(), 1)) / (grad_w.shape[0] * grad_w.shape[1]) * 0.01
                    result = (torch.zeros(grad_b.shape).cuda(), grad_input[1], grad_input[2])
                else:
                    # use 'sp' to calculate new grad_w
                    grad_b_new = torch.zeros(grad_b.shape).cuda()
                    tp = torch.mm(torch.mm(torch.mm(sp, module.rand), module.error.t()), module.input).t()
                    hebb_use = torch.mm(module.hebb_delta, sp) * 1e-2
                    grad_w_new = tp + hebb_use
                    result = (grad_b_new, grad_input[1], grad_w_new)

                    if bp_mark and (counter == bp_mark):
                        result = (torch.zeros(grad_b.shape).cuda(), grad_input[1], grad_input[2])
                        counter = 0

                return result

            if sbp:
                for name, module in self.layers[i].named_children():
                    if name is 'fc':
                        module.register_backward_hook(hook_fc_backward)

    def forward(self, input, labels=None):
        for step in range(self.hyperparams[4]):
            if self.hyperparams[5] == 'MNIST':
                x = input > torch.rand(input.size()).cuda()
            elif self.hyperparams[5] == 'NETTalk':
                x = input.cuda()
            elif self.hyperparams[5] == 'DVSGesture':
                x = input[:, 0 + step * 5, :]
            x = x.float()
            x = x.view(self.hyperparams[0], -1)
            for i in range(self.len):
                x = self.layers[i](x)
                if i != 0:
                    self.layers[i].h_last = self.layers[i - 1].h_now

        outputs = self.layers[-1].sumspike / self.hyperparams[4]

        if labels is not None:
            error = labels - outputs

            for i in range(self.len):
                self.layers[i].fc.error = error
                if i != self.len - 1:
                    self.layers[i].fc.hebb_delta = self.layers[i + 1].hebb_last[-1] + self.layers[i + 1].hebb_last[-2] - self.layers[i + 1].hebb_last[0] - self.layers[i + 1].hebb_last[1]

        potential = 0

        for i in range(self.len - 1):
            l_potential = torch.mm(self.layers[i + 1].h_now, torch.mm(self.layers[i].fc.weight.data, self.layers[i].h_now.t())).mean() / float(self.layers[i].input_size * self.layers[i].output_size) + torch.mm(self.layers[i + 1].h_now, self.layers[i + 1].h_now.t()).mean() / float(self.layers[i].output_size * self.layers[i].output_size)
            potential = potential + l_potential.abs()

        l_potential = torch.mm(x, torch.mm(self.layers[-1].fc.weight.data, self.layers[-1].h_now.t())).mean() / float(self.layers[-1].input_size * self.layers[-1].output_size) + torch.mm(x, x.t()).mean() / float(self.layers[-1].output_size * self.layers[-1].output_size)
        potential = potential + l_potential.abs()

        return outputs, potential