class InceptionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.br1 = nn.Conv2d(16, 8, 1)
        self.br2_reduce = nn.Conv2d(16, 8, 1)
        self.br2_conv = nn.Conv2d(8, 12, 3, padding=1)
        self.br3_reduce = nn.Conv2d(16, 4, 1)
        self.br3_conv1 = nn.Conv2d(4, 6, 3, padding=1)
        self.br3_conv2 = nn.Conv2d(6, 6, 3, padding=1)
        self.br4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.br4_proj = nn.Conv2d(16, 6, 1)

    def forward(self, x):
        a = torch.relu(self.br1(x))
        b = torch.relu(self.br2_conv(torch.relu(self.br2_reduce(x))))
        c = torch.relu(self.br3_reduce(x))
        c = torch.relu(self.br3_conv2(torch.relu(self.br3_conv1(c))))
        d = torch.relu(self.br4_proj(self.br4_pool(x)))
        return torch.cat([a, b, c, d], dim=1)
