class DeconvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.dec = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.dec(torch.relu(self.enc(x))))
