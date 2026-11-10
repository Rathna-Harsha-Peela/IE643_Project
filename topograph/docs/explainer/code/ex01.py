class TierL0_ToyMLP_201(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(3, 1)
        self.fc1_u2 = nn.Linear(2, 1)
        self.unused_op_35 = nn.Identity()
        self.fc1_u0 = nn.Linear(2, 1)
        self.fc1_u1 = nn.Linear(2, 1)

    def forward(self, inp_tensor):
        u0 = torch.relu(self.fc1_u0(inp_tensor))
        u1 = torch.relu(self.fc1_u1(inp_tensor))
        u2 = torch.relu(self.fc1_u2(inp_tensor))
        h = torch.cat([u0, u1, u2], dim=1)
        y = self.out(h)
        _dead_val = self.unused_op_35(inp_tensor)
        return y