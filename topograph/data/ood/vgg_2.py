class GrayClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1a = nn.Conv2d(1, 12, 3, padding=1)
        self.c1b = nn.Conv2d(12, 12, 3, padding=1)
        self.down1 = nn.AvgPool2d(2)
        self.c2a = nn.Conv2d(12, 24, 3, padding=1)
        self.down2 = nn.AvgPool2d(4)
        self.out_fc = nn.Linear(24 * 2 * 2, 7)

    def forward(self, image):
        h = torch.relu(self.c1b(torch.relu(self.c1a(image))))
        h = self.down1(h)
        h = self.down2(torch.relu(self.c2a(h)))
        return self.out_fc(h.flatten(1))
