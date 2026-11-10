class StackSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Conv2d(4, 4, 1)
        self.p2 = nn.Conv2d(4, 4, 3, padding=1)
        self.p3 = nn.Conv2d(4, 4, 5, padding=2)

    def forward(self, x):
        s = torch.stack([self.p1(x), self.p2(x), self.p3(x)], dim=0)
        return s.sum(0)
