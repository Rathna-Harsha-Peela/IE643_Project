class TierL0_ToyMLP_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_u0 = nn.Linear(3, 1)
        self.fc1_u1 = nn.Linear(3, 1)
        self.out = nn.Linear(2, 2)

    def forward(self, x):
        u0 = torch.relu(self.fc1_u0(x))
        u1 = torch.relu(self.fc1_u1(x))
        h = torch.cat([u0, u1], dim=1)
        y = self.out(h)
        return y