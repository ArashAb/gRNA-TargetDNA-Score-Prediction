import torch.nn as nn
import torch

class RegressionModel(nn.Module):
    def __init__(self, input_dim=320 * 2, hidden_dim=512):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)
