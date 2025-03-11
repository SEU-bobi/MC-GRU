import torch
import torch.nn as nn
from torch.autograd import Variable

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device('cpu')

class MC_GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MC_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.linear_r = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.linear_h = nn.Linear(input_size + hidden_size + 2, hidden_size, bias=True)
        self.linear_k = nn.Linear(1 + hidden_size, 1, bias=True)
        self.linear_m = nn.Linear(input_size + hidden_size, 1, bias=True)

    def forward(self, input, hidden, K_State, St, Mass):
        combined = torch.cat((input, hidden), 1)
        z = torch.sigmoid(self.linear_z(combined))
        r = torch.sigmoid(self.linear_r(combined))
        h_tilde = torch.tanh(self.linear_h(torch.cat((input, hidden * r, St * K_State, Mass), 1)))
        new_hidden = (1 - z) * hidden + z * h_tilde
        new_K_State = torch.sigmoid(self.linear_k(torch.cat((new_hidden, K_State), dim=1)))
        return new_hidden, new_K_State

    def loop(self, inputs, St, Mass):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)

        Hidden_States = []
        Hidden_State, K_State = self.initHidden(batch_size)
        for i in range(time_step):
            # if i == 0:
            #     St = torch.zeros(St.shape).to(device)
            #     Mass = torch.zeros(Mass.shape).to(device)
            Hidden_State, K_State = self.forward(inputs[:, i:i + 1, :].squeeze(1), Hidden_State, K_State, St, Mass)
            Hidden_States.append(Hidden_State)
        Hidden_States = torch.stack(Hidden_States, dim=1)
        return Hidden_States

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(device))
        K_State = Variable(torch.zeros(batch_size, 1).to(device))
        return Hidden_State, K_State


class MC_Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MC_Net, self).__init__()
        self.rnn1 = nn.GRU(input_size=input_size, hidden_size=64).to(device)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=False)
        self.rnn2 = MC_GRU(input_size=64, hidden_size=64).to(device)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, init_k, Mass):
        f, _ = self.rnn1(x)
        f = self.relu(f)
        k = self.rnn2.loop(f, init_k, Mass)
        k = self.relu(k)
        y = f + k
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return y


#
net = MC_Net(input_size=1, num_classes=1)
batch_size = 10
seq_len = 1000
K = torch.randn(batch_size, 1)
M = torch.randn(batch_size, 1)
input = torch.randn(batch_size, seq_len, 1)
out = net(input, K, M)
print(out.shape)
