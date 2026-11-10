class SEResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(8, 8, 3, padding=1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite_down = nn.Conv2d(8, 2, 1)
        self.excite_up = nn.Conv2d(2, 8, 1)

    def forward(self, feat):
        b = self.body(feat)
        w = self.squeeze(b)
        w = torch.sigmoid(self.excite_up(F.relu(self.excite_down(w))))
        return F.relu(b * w + feat)
