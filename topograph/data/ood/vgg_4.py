class StridedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.s2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(32 * 2 * 2, 4)

    def forward(self, x):
        x = F.relu(self.s1(x))
        x = F.relu(self.s2(x))
        x = self.pool(x)
        x = self.drop(torch.flatten(x, 1))
        return self.linear(x)
