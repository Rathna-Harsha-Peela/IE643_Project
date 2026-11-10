class DynamicPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        k = int(y.shape[2]) // 2
        return F.max_pool2d(y, k)
