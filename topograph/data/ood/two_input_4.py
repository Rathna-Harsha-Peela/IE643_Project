class CrossFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_conv = nn.Conv2d(4, 8, 3, padding=1)
        self.depth_conv = nn.Conv2d(4, 8, 3, padding=1)
        self.mix = nn.Conv2d(16, 8, 1)
        self.refine = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, rgb, depth):
        a = F.relu(self.rgb_conv(rgb))
        b = F.relu(self.depth_conv(depth))
        m = F.relu(self.mix(torch.cat([a, b], dim=1)))
        return self.refine(m) + a
