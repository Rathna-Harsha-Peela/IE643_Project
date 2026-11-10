class FuncPoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.out = nn.Linear(16 * 4 * 4, 5)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.c1(x)), 2)
        x = F.avg_pool2d(torch.relu(self.c2(x)), 2)
        x = x.view(1, -1)
        return self.out(x)
