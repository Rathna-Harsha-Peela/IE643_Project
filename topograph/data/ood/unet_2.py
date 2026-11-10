class SegNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_conv = nn.Conv2d(1, 6, 3, padding=1)
        self.down_pool = nn.MaxPool2d(2)
        self.bottom = nn.Conv2d(6, 12, 3, padding=1)
        self.up_conv = nn.Conv2d(18, 6, 3, padding=1)
        self.seg_head = nn.Conv2d(6, 2, 1)

    def forward(self, img):
        skip = F.relu(self.down_conv(img))
        b = F.relu(self.bottom(self.down_pool(skip)))
        b = F.interpolate(b, scale_factor=2.0, mode="nearest")
        u = F.relu(self.up_conv(torch.cat([b, skip], dim=1)))
        return self.seg_head(u)
