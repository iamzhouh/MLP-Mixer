import torch
import numpy as np
from einops.layers.torch import Rearrange   # 使用einops中的Rearrange实现矩阵旋转
from torchsummary import summary   # 使用torchsummary来查看网络结构
import torch.nn.functional as F



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),    #  probability of an element to be zeroed. Default: 0.5
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.net(x)
        return x

#测试多层感知机
# mlp=FeedForward(10,20,0.4).to(device)
# summary(mlp,input_size=(10,))

class MixerBlock(torch.nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super(MixerBlock, self).__init__()
        self.token_mixer = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),   # LN  标准化
            Rearrange('b n d -> b d n'),  # 旋转 这里是[batch_size, num_patch, dim] -> [batch_size, dim, num_patch]
            FeedForward(num_patch, token_dim, dropout),   # MLP1
            Rearrange('b d n -> b n d')   # 旋转
        )

        self.channel_mixer = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),   # LN  标准化
            FeedForward(dim, channel_dim, dropout)    # MLP1
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)

        return x

# 测试mixerblock
# x = torch.randn(1, 196, 512)
# mixer_block = MixerBlock(512, 196, 32, 32)
#
# x = mixer_block(x)
# print(x.shape)

class MLPMixer(torch.nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim, dropout = 0.):
        super(MLPMixer, self).__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size)**2

        # embedding 操作，用卷积来分成一小块一小块的
        self.to_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        self.mixer_blocks = torch.nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(
                MixerBlock(dim, self.num_patches, token_dim, channel_dim, dropout)
            )

        self.layer_normal = torch.nn.LayerNorm(dim)

        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes)
        )


    def forward(self, x):
        x = self.to_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_normal(x)
        x = x.mean(dim = 1)

        x =self.mlp_head(x)

        return x

# 测试MLP-Mixer
# if __name__ == '__main__':
#     model = MLPMixer(in_channels=3, dim=512, num_classes=1000, patch_size=16, image_size=224, depth=1, token_dim=256, channel_dim=2048).to(device)
#     summary(model, input_size = (3, 224, 224))
