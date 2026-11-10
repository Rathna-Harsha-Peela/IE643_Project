class ConvStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1),
                                     nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1)])
        self.proj = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        for blk in self.blocks:
            x = F.relu(blk(x)) + x
        return self.proj(x)
