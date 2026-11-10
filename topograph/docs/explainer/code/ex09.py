class SEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.fc1 = nn.Linear(16, 4)
        self.fc2 = nn.Linear(4, 16)

    def forward(self, x):
        y = torch.relu(self.conv(x))
        s = y.mean((2, 3))
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        s = s.unsqueeze(-1).unsqueeze(-1)
        return y * s
