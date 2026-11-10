class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(12, 6)
        self.b = nn.Linear(12, 6)
        self.sm = nn.Softmax(dim=1)

    def forward(self, z):
        s = self.a(z) * self.b(z)
        return self.sm(s)
