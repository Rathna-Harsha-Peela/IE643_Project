class CheckedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        assert h.sum() >= 0
        return self.fc2(h)
