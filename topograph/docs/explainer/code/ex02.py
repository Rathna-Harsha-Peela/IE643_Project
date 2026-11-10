class TierL1_LinearChain_222(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused_op_77 = nn.Identity()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.act0 = nn.GELU()
        self.conv1 = nn.Conv2d(16, 16, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = torch.relu(x)
        _dead_val = self.unused_op_77(x)
        return x