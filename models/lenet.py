# Code adapted from https://github.com/bolianchen/Data-Free-Learning-of-Student-Networks

#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch.nn as nn
import torch
from torch.autograd import Variable
use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

class LeNet5_(nn.Module):

    def __init__(self, in_channels):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(),
        )
        self. classifier = nn.Sequential(
            nn.Linear(20, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.fc_mu = nn.Linear(120, 20)
        self.fc_var = nn.Linear(120, 20)

    def forward(self, img, out_feature=False, out_activation=False):
        feature = self.feature_extractor(img)
        feature = feature.view(-1, 120)
        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)
        z = self.reparameterize(mu, log_var)
        output = self.classifier(z)
        if out_feature == False:
            return output
        else:
            if out_activation == True:
                return output, feature
            else:
                return output, z #mu, log_var #,feature

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space.
        In order for the back-propagation to work, we need to be able to calculate the gradient.
        This reparameterization trick first generates a normal distribution, then shapes the distribution
        with the mu and variance from the encoder.

        This way, we can can calculate the gradient parameterized by this particular random instance.
        """
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_()).to(device)
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    # question: how is the loss function using the mu and variance?
    def forward(self, mu, log_var):
        """gives the batch normalized Variational Error."""

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return KLD

class LeNet5(nn.Module):

    def __init__(self, in_channels):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img, out_feature=False, out_activation=False):
        activation1 = self.conv1(img)
        output = self.relu1(activation1)
        output = self.maxpool1(output)
        activation2 = self.conv2(output)
        output = self.relu2(activation2)
        output = self.maxpool2(output)
        activation3 = self.conv3(output)
        output = self.relu3(activation3)
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            if out_activation == True:
                return output, feature, activation1, activation2, activation3
            else:
                return output, feature


class LeNet5Half(nn.Module):

    def __init__(self, in_channels):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 42)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(42, 10)

    def forward(self, img, out_feature=False, out_activation=False):  # [batch_size, 1, 32, 32]
        activation1 = self.conv1(img)  # [batch_size, 3, 28, 28]
        output = self.relu1(activation1)  # [batch_size, 3, 28, 28]
        output = self.maxpool1(output)  # [batch_size, 3, 14, 14]
        activation2 = self.conv2(output)  # [batch_size, 8, 10, 10]
        output = self.relu2(activation2)  # [batch_size, 8, 10, 10]
        output = self.maxpool2(output)  # [batch_size, 8, 5, 5]
        activation3 = self.conv3(output)  # [batch_size, 60, 1, 1]
        output = self.relu3(activation3)  # [batch_size, 60, 1, 1]
        feature = output.view(-1, 60)  # [batch_size, 60]
        output = self.fc1(feature)  # [batch_size, 42]
        output = self.relu4(output)
        output = self.fc2(output)  # [batch_size, 10]
        if out_feature == False:
            return output
        else:
            if out_activation == True:
                return output, feature, activation1, activation2, activation3
            else:
                return output, feature
