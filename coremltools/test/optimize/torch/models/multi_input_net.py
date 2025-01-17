#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch.nn as nn

num_classes = 10


class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_branch1 = nn.Conv2d(1, 32, (5, 5), padding=2)
        self.relu1_branch1 = nn.ReLU()
        self.pool1_branch1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn1_branch1 = nn.BatchNorm2d(32, eps=0.001, momentum=0.01)

        self.conv1_branch2 = nn.Conv2d(1, 32, (5, 5), padding=2)
        self.relu1_branch2 = nn.ReLU()
        self.pool1_branch2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn1_branch2 = nn.BatchNorm2d(32, eps=0.001, momentum=0.01)

        self.conv2 = nn.Conv2d(32, 64, (5, 5), padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(3136, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.dense2 = nn.Linear(1024, num_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, input1, input2):
        x_branch1 = self.conv1_branch1(input1)
        x_branch1 = self.relu1_branch1(x_branch1)
        x_branch1 = self.pool1_branch1(x_branch1)
        x_branch1 = self.bn1_branch1(x_branch1)

        x_branch2 = self.conv1_branch2(input2)
        x_branch2 = self.relu1_branch2(x_branch2)
        x_branch2 = self.pool1_branch2(x_branch2)
        x_branch2 = self.bn1_branch2(x_branch2)

        # Combine branches
        x = x_branch1 + x_branch2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x
