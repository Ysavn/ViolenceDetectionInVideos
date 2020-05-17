import torch
from torch import nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        torch.nn.init.xavier_normal(self.Gates.weight)
        torch.nn.init.constant(self.Gates.bias, 0)

    def forward(self, input_, prev_state):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (Variable(torch.zeros(state_size).to(self.device)),
                Variable(torch.zeros(state_size).to(self.device)))

        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell
