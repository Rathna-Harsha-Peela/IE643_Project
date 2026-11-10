class IterativeRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Linear(8, 8)

    def forward(self, x):
        while x.norm() > 1.0:
            x = torch.tanh(self.step(x)) * 0.5
        return x
