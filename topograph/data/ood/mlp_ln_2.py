class TabularNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_norm = nn.LayerNorm(20)
        self.hidden_a = nn.Linear(20, 64)
        self.gelu_a = nn.GELU()
        self.hidden_b = nn.Linear(64, 64)
        self.gelu_b = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(64, 3)

    def forward(self, x):
        x = self.inp_norm(x)
        x = self.gelu_a(self.hidden_a(x))
        x = self.dropout(self.gelu_b(self.hidden_b(x)))
        return self.classifier(x)
