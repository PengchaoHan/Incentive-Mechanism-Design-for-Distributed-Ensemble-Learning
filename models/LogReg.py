import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset_data_files')))
import torch.nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, output_dim=10):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        outputs = self.linear(x)
        return outputs