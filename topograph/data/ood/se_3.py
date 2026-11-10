class ChannelGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(12, 12, 3, padding=1)
        self.gate_fc1 = nn.Linear(12, 3)
        self.gate_fc2 = nn.Linear(3, 12)
        self.out_conv = nn.Conv2d(12, 6, 1)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        pooled = x.mean((2, 3))
        g = torch.sigmoid(self.gate_fc2(torch.relu(self.gate_fc1(pooled))))
        x = x * g.unsqueeze(-1).unsqueeze(-1)
        return self.out_conv(x)
