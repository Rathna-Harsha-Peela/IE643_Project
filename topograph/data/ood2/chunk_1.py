class GLUConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(8, 16, 3, padding=1)
        self.proj = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        h = self.expand(x)
        a, b = torch.chunk(h, 2, dim=1)
        g = a * torch.sigmoid(b)
        return self.proj(g)
