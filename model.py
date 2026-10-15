import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import hparams




class NatureCNN(nn.Module):
    def __init__(self, action_space_n):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*7*7, out_features=512)
        self.relu = nn.ReLU()

        self.policy = nn.Linear(in_features=512, out_features=action_space_n)
        self.value = nn.Linear(in_features=512, out_features=1)

        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(layer.weight, nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.fc.weight, nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.policy.weight, hparams.policy_gain)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, hparams.value_gain)
        nn.init.zeros_(self.value.bias)


    def forward(self, x):
        x = x.float() / 255.0

        x = self.conv1(x)       # 32, 20, 20
        x = self.relu(x)

        x = self.conv2(x)       # 64, 9, 9
        x = self.relu(x)

        x = self.conv3(x)       # 64, 7, 7
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)

        return x

    def get_value(self, x):
        x = self(x)
        return self.value(x)

    def get_policy_and_value(self, x, action=None):
        x = self(x)
        logits = self.policy(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.value(x)