class ELUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 16)
        self.l2 = nn.Linear(16, 16)
        self.act = nn.ELU()
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h)) + x
        return self.head(h)
