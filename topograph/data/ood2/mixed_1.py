class MixedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        a = F.gelu(self.c1(x))
        b = F.silu(self.c2(a)) + a
        p = F.adaptive_avg_pool2d(b, 1).flatten(1)
        return F.log_softmax(self.fc(p), dim=1)
