class TwoStageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(1, 4, 3, stride=2, padding=1)
        self.e2 = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.d2 = nn.ConvTranspose2d(8, 1, 2, stride=2)

    def forward(self, x):
        a = torch.relu(self.e1(x))
        b = torch.relu(self.e2(a))
        c = torch.relu(self.d1(b))
        return self.d2(torch.cat([c, a], dim=1))
