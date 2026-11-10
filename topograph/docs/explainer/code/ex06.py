class TierL0_ToyMLP_215(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_u0 = nn.Linear(3, 1)
        self.fc1_u1 = nn.Linear(3, 1)
        self.fc1_u2 = nn.Linear(3, 1)
        self.fc1_u3 = nn.Linear(3, 1)
        self.fc2 = nn.Linear(4, 2)
        self.out = nn.Linear(2, 2)

    def forward(self, inp_tensor):
        u0 = torch.relu(self.fc1_u0(inp_tensor))
        u1 = torch.relu(self.fc1_u1(inp_tensor))
        u2 = torch.relu(self.fc1_u2(inp_tensor))
        u3 = torch.relu(self.fc1_u3(inp_tensor))
        h = torch.cat([u0, u1, u2, u3], dim=1)
        h = torch.relu(self.fc2(h))
        y = self.out(h)
        return y