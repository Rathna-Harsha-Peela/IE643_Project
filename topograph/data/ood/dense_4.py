class FourLayerDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(10, 5, 1)
        self.d2 = nn.Conv2d(15, 5, 1)
        self.d3 = nn.Conv2d(20, 5, 1)
        self.d4 = nn.Conv2d(25, 5, 1)

    def forward(self, x):
        a = torch.relu(self.d1(x))
        b = torch.relu(self.d2(torch.cat([x, a], dim=1)))
        c = torch.relu(self.d3(torch.cat([x, a, b], dim=1)))
        d = torch.relu(self.d4(torch.cat([x, a, b, c], dim=1)))
        return d
