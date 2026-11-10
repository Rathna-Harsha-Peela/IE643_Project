class SplitMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Conv2d(6, 12, 1)
        self.left = nn.Conv2d(6, 6, 3, padding=1)
        self.right = nn.Conv2d(6, 6, 5, padding=2)

    def forward(self, x):
        p, q = torch.split(self.pre(x), 6, dim=1)
        return torch.cat([self.left(p), self.right(q)], dim=1) + self.pre(x)
