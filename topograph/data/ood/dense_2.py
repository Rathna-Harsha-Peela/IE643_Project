class DenseNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_a = nn.Conv2d(6, 6, 3, padding=1)
        self.layer_b = nn.Conv2d(12, 6, 3, padding=1)
        self.transition = nn.Conv2d(18, 9, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, inp):
        fa = F.relu(self.layer_a(inp))
        fb = F.relu(self.layer_b(torch.cat([inp, fa], dim=1)))
        t = self.transition(torch.cat([inp, fa, fb], dim=1))
        return self.pool(t)
