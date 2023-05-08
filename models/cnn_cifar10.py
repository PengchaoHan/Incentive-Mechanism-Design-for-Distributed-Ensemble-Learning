import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset_data_files')))
import torch
import torch.nn as nn


class ModelCNNCifar10(nn.Module):
    def __init__(self):
        super(ModelCNNCifar10, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)  # -------
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)  # -------
        )
        self.fc2 = nn.Linear(128, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x):
        conv_ = self.conv(x)
        fc_ = conv_.view(-1, 8 * 8 * 64)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        return output


class ModelCNNCifar10_small(nn.Module):
    def __init__(self):
        super(ModelCNNCifar10_small, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)  # -------
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5)  # -------
        )
        self.fc2 = nn.Linear(256, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x, out_activation=False):
        conv_ = self.conv(x)
        fc_ = conv_.view(-1, 32*8*8)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        if out_activation:
            return output, conv_
        else:
            return output

