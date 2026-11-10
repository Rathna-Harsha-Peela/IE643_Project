class WideDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(6, 5)
        self.deep1 = nn.Linear(6, 24)
        self.deep_norm = nn.LayerNorm(24)
        self.deep2 = nn.Linear(24, 5)
        self.drop = nn.Dropout(0.3)

    def forward(self, inputs):
        w = self.wide(inputs)
        d = torch.relu(self.deep_norm(self.deep1(inputs)))
        d = self.deep2(self.drop(d))
        return torch.sigmoid(w + d)
