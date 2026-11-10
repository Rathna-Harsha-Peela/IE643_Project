class StackedBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_in = nn.Conv2d(8, 4, 1)
        self.a_mid = nn.Conv2d(4, 4, 3, padding=1)
        self.a_out = nn.Conv2d(4, 16, 1)
        self.a_proj = nn.Conv2d(8, 16, 1)
        self.b_in = nn.Conv2d(16, 4, 1)
        self.b_mid = nn.Conv2d(4, 4, 3, padding=1)
        self.b_out = nn.Conv2d(4, 16, 1)

    def forward(self, x):
        h = self.a_out(F.relu(self.a_mid(F.relu(self.a_in(x)))))
        x = F.relu(h + self.a_proj(x))
        h = self.b_out(F.relu(self.b_mid(F.relu(self.b_in(x)))))
        return F.relu(h + x)
