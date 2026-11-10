class ReshapeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(4, 4, 3, padding=1)
        self.fc = nn.Linear(64, 3)

    def forward(self, img):
        y = F.relu(self.body(img))
        y = F.max_pool2d(y, 4)
        y = y.reshape(1, 64)
        return self.fc(y)
