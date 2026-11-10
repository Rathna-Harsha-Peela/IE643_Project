class ImageMetaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.meta_fc = nn.Linear(5, 8)
        self.head = nn.Linear(16, 2)

    def forward(self, img, meta):
        a = torch.flatten(self.pool(torch.relu(self.conv(img))), 1)
        b = torch.relu(self.meta_fc(meta))
        return self.head(torch.cat([a, b], dim=1))
