class EnsembleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(8, 4)
        self.h2 = nn.Linear(8, 4)
        self.h3 = nn.Linear(8, 4)

    def forward(self, x):
        votes = torch.stack([self.h1(x), self.h2(x), self.h3(x)], dim=1)
        return votes.mean(1)
