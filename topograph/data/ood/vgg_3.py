class VGGBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(4, 16, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(16)
        self.pool_a = nn.MaxPool2d(2)
        self.conv_b = nn.Conv2d(16, 32, 3, padding=1)
        self.bn_b = nn.BatchNorm2d(32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        x = self.pool_a(F.relu(self.bn_a(self.conv_a(x))))
        x = F.relu(self.bn_b(self.conv_b(x)))
        x = torch.flatten(self.gap(x), 1)
        return self.fc2(F.relu(self.fc1(x)))
