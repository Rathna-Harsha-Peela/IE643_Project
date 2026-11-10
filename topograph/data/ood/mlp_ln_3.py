class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 16)
        self.ln = nn.LayerNorm(16)
        self.ff1 = nn.Linear(16, 48)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(48, 16)
        self.head = nn.Linear(16, 2)

    def forward(self, tokens):
        e = self.embed(tokens)
        y = self.ff2(self.act(self.ff1(self.ln(e))))
        return self.head(e + y)
