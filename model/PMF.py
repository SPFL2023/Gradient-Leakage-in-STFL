import torch.nn as nn
import torch


class PMFModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(PMFModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(977, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_code,):

        h0 = torch.zeros(1, 1, self.hidden_size).cuda()
        c0 = torch.zeros(1, 1, self.hidden_size).cuda()

        out, (ho, co) = self.lstm(input_code, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out