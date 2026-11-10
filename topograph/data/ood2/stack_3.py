class AddNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 3, 3, padding=1)
        self.d = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return torch.add(self.c(x), self.d(x))
