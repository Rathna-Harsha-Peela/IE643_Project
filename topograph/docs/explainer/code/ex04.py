class TierL3_MultiBranch_202(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0_c0 = nn.Conv2d(12, 8, kernel_size=5, padding=2)
        self.b1_c0 = nn.Conv2d(12, 8, kernel_size=1, padding=0)
        self.b1_c1 = nn.Conv2d(8, 8, kernel_size=1)
        self.b2_c0 = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.head = nn.Conv2d(24, 16, kernel_size=1)

    def forward(self, x):
        b0 = torch.relu(self.b0_c0(x))
        b1 = torch.relu(self.b1_c0(x))
        b1 = torch.relu(self.b1_c1(b1))
        b2 = torch.relu(self.b2_c0(x))
        merged = torch.cat([b0, b1, b2], dim=1)
        out = self.head(merged)
        return out