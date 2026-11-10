class ThreeWay(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Conv2d(8, 4, 1)
        self.p2a = nn.Conv2d(8, 4, 1)
        self.p2b = nn.Conv2d(4, 4, 5, padding=2)
        self.p3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.fuse = nn.Conv2d(16, 8, 1)

    def forward(self, inp):
        o1 = self.p1(inp)
        o2 = self.p2b(F.relu(self.p2a(inp)))
        o3 = self.p3(inp)
        merged = torch.cat([o1, o2, o3], dim=1)
        return F.relu(self.fuse(merged))
