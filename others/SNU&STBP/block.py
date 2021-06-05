import torch
import torch.nn as nn

thresh, lens, decay, pseudo = (0.5, 0.3, 0.2, 'STBP')

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        if pseudo == 'STBP':
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
            return grad_input * temp.float()

        elif pseudo == 'SNU':
            z = ctx.saved_tensors
            s = torch.sigmoid(z[0])
            return (1 - s) * s * grad_h

act_fun = ActFun.apply

class LinearPlus(nn.Linear):
    def __init__(self, input_size, output_size, bias, last_layer,rand):
        super(LinearPlus, self).__init__(input_size, output_size, bias)
        self.last_layer = last_layer
        self.rand = rand
        self.input = None
        self.error = None
        self.hebb_delta = None

class FC_block(nn.Module):
    def __init__(self, hyperparams, input_size, output_size, nextlayer_size, if_bias = True):
        super(FC_block, self).__init__()
        self.batch_size = hyperparams[0]
        self.input_size = input_size
        self.output_size = output_size
        self.nextlayer_size = nextlayer_size
        self.last_layer = (output_size == hyperparams[2])
        self.h_last = None
        self.h_now = None
        self.hebb_last = []
        self.hebb_delta = None
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0

        rand = None
        if not self.last_layer:
            rand = nn.Parameter(torch.Tensor(torch.Size([self.output_size, hyperparams[2]])))
            torch.nn.init.kaiming_uniform_(rand)
            rand.requires_grad = False

        self.fc = LinearPlus(self.input_size, self.output_size, if_bias, self.last_layer, rand)

    def mem_update(self, x):
        I = torch.sigmoid(self.fc(x)) - 0.1
        self.mem = self.mem * decay * (1 - self.spike) + I
        self.spike = act_fun(self.mem)
        self.sumspike = self.sumspike + self.spike

    def forward(self, input):
        if self.time_counter == 0:
            self.mem = torch.zeros((self.batch_size, self.output_size)).cuda()
            self.spike = torch.zeros((self.batch_size, self.output_size)).cuda()
            self.sumspike = torch.zeros((self.batch_size, self.output_size)).cuda()
            self.hebb_last = []

        self.time_counter += 1
        self.h_now = input
        self.fc.input = input
        self.mem_update(input)

        if self.h_last is not None:
            self.hebb_last.append(torch.mm(self.h_last.t(), self.h_now) / (self.batch_size * self.batch_size))

        if self.time_counter == 20:
            self.time_counter = 0

        output = self.spike

        return output