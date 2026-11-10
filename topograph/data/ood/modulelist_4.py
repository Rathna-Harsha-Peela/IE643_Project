class IndexedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(6, 12), nn.Linear(12, 12), nn.Linear(12, 6)])
        self.norm = nn.LayerNorm(6)

    def forward(self, x):
        h = torch.tanh(self.hidden[0](x))
        h = torch.tanh(self.hidden[1](h))
        h = self.hidden[2](h)
        return self.norm(h + x)
