class DropMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(12, 24)
        self.b = nn.Linear(24, 12)

    def forward(self, x):
        h = F.dropout(F.leaky_relu(self.a(x), 0.2), p=0.1, training=self.training)
        return F.normalize(self.b(h) + x, dim=1)
