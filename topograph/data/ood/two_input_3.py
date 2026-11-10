class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_tower = nn.Linear(10, 16)
        self.item_tower = nn.Linear(6, 16)
        self.user_norm = nn.LayerNorm(16)
        self.item_norm = nn.LayerNorm(16)
        self.scorer = nn.Linear(16, 1)

    def forward(self, user, item):
        u = self.user_norm(torch.relu(self.user_tower(user)))
        i = self.item_norm(torch.relu(self.item_tower(item)))
        return self.scorer(u * i)
