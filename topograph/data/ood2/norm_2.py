class HardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(6, 12, 1)
        self.act_a = nn.Hardswish()
        self.b = nn.Conv2d(12, 6, 3, padding=1)
        self.act_b = nn.ELU()

    def forward(self, x):
        y = self.act_a(self.a(x))
        return self.act_b(self.b(y) + x)
