class AffineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(5, 5)
        self.o = nn.Linear(5, 1)

    def forward(self, x):
        h = torch.tanh(self.l(x)) * 2.0 - 1.0
        return self.o(h + x)
