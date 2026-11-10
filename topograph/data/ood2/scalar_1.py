class ScaledResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Conv2d(6, 6, 3, padding=1)

    def forward(self, x):
        return x + 0.1 * self.f(x)
