class EarlyExit(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Conv2d(3, 8, 3, padding=1)
        self.exit_head = nn.Conv2d(8, 2, 1)
        self.stage2 = nn.Conv2d(8, 8, 3, padding=1)
        self.final_head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        h = torch.relu(self.stage1(x))
        logits = self.exit_head(h)
        if logits.softmax(1).max() > 0.9:
            return logits
        h = torch.relu(self.stage2(h))
        return self.final_head(h)
