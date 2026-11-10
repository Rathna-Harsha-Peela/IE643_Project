class GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(10, 32)
        self.out = nn.Linear(16, 2)

    def forward(self, feats):
        h = self.inp(feats)
        u, v = h.chunk(2, dim=1)
        return self.out(u * torch.tanh(v))
