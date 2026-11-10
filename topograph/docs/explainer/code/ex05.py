class TierL4_Nested_201(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pa_conv0 = nn.Conv2d(16, 8, kernel_size=1, padding=0)
        self.pa_conv1 = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.pa_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pb_conv0 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pb_conv1 = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.merge = nn.Conv2d(16, 16, kernel_size=1)
        self.head = nn.Conv2d(16, 8, kernel_size=1)

    def forward(self, x):
        stem_out = torch.relu(self.stem(x))
        pa = stem_out
        pb = stem_out
        pa = torch.relu(self.pa_conv0(pa))
        pa = torch.relu(self.pa_conv1(pa))
        pa = torch.relu(self.pa_conv2(pa))
        pb = torch.relu(self.pb_conv0(pb))
        pb = torch.relu(self.pb_conv1(pb))
        cat_ab = torch.cat([pa, pb], dim=1)
        merged = self.merge(cat_ab)
        combined = torch.relu(merged + stem_out)
        out = self.head(combined)
        return out