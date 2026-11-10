class RepeatByInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x):
        n = int(x.abs().sum().item()) % 3 + 1
        for _ in range(n):
            x = torch.relu(self.block(x))
        return x
