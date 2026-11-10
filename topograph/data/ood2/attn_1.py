class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k = nn.Linear(8, 8)
        self.v = nn.Linear(8, 8)
        self.o = nn.Linear(8, 4)

    def forward(self, x):
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(1)
        w = torch.softmax(torch.matmul(q, k), dim=-1)
        y = torch.matmul(w, self.v(x).unsqueeze(2)).squeeze(2)
        return self.o(y)
