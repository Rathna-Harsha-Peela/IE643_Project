class TierL2_Residual_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0_bn1 = nn.BatchNorm2d(16)
        self.b1_conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.b0_bn2 = nn.BatchNorm2d(32)
        self.b1_proj = nn.Conv2d(32, 32, kernel_size=1)
        self.b1_bn1 = nn.BatchNorm2d(16)
        self.b0_conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.b0_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.b0_proj = nn.Conv2d(32, 32, kernel_size=1)
        self.b1_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.b1_bn2 = nn.BatchNorm2d(32)
        self.unused_op_37 = nn.Identity()

    def forward(self, feat_map):
        res_0 = feat_map
        out_0 = torch.relu(self.b0_bn1(self.b0_conv1(feat_map)))
        out_0 = self.b0_bn2(self.b0_conv2(out_0))
        res_0 = self.b0_proj(res_0)
        feat_map = torch.relu(out_0 + res_0)
        res_1 = feat_map
        out_1 = torch.relu(self.b1_bn1(self.b1_conv1(feat_map)))
        out_1 = self.b1_bn2(self.b1_conv2(out_1))
        res_1 = self.b1_proj(res_1)
        feat_map = torch.relu(out_1 + res_1)
        _dead_val = self.unused_op_37(feat_map)
        return feat_map