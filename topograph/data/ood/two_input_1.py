class SiameseFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_left = nn.Conv2d(3, 8, 3, padding=1)
        self.enc_right = nn.Conv2d(3, 8, 3, padding=1)
        self.fuse = nn.Conv2d(8, 4, 1)

    def forward(self, left, right):
        l = torch.relu(self.enc_left(left))
        r = torch.relu(self.enc_right(right))
        return self.fuse(l + r)
