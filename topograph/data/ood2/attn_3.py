class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Conv2d(4, 1, 1)
        self.val = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        w = torch.softmax(self.key(x).flatten(2), dim=-1)
        v = self.val(x).flatten(2)
        ctx = torch.matmul(v, w.transpose(1, 2))
        return x + ctx.unsqueeze(-1)
