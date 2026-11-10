class InceptionRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Conv2d(24, 8, 1)
        self.branch_b1 = nn.Conv2d(24, 8, 1)
        self.branch_b2 = nn.Conv2d(8, 8, 3, padding=1)
        self.branch_c1 = nn.Conv2d(24, 8, 1)
        self.branch_c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.branch_c3 = nn.Conv2d(8, 8, 3, padding=1)
        self.mix = nn.Conv2d(24, 24, 1)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b2(self.branch_b1(x))
        c = self.branch_c3(self.branch_c2(self.branch_c1(x)))
        m = self.mix(torch.cat([a, b, c], dim=1))
        return torch.relu(x + m)
