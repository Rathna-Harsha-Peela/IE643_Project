class MLPNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(12, 32)
        self.norm1 = nn.LayerNorm(32)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.fc_out = nn.Linear(32, 4)

    def forward(self, features):
        h = self.act1(self.norm1(self.fc_in(features)))
        h = self.drop(h)
        return self.fc_out(h)
