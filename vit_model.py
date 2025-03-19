import numpy as np
import torch
import torch.nn as nn

# https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # normalize
            nn.Linear(dim, hidden_dim),  # dense
            nn.GELU(),  # activation
            nn.Dropout(dropout),  # generalization
            nn.Linear(hidden_dim, dim),  # dense
            nn.Dropout(dropout),  # generalization
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads  # number of qkv
        project_out = not (heads == 1 and dim_head == dim
                           )  # no need to recombine the output

        self.heads = heads
        self.scale = dim_head**-0.5  # 1/sqrt(q)

        self.norm = nn.LayerNorm(dim)  # normalize, LayerNorm | BatchNorm

        self.attend = nn.Softmax(dim=-1)  # calculate probability
        self.dropout = nn.Dropout(dropout)  # generalization

        self.to_qkv = nn.Linear(
            dim, inner_dim * 3,
            bias=False)  # number of qkv * 3, later split into q, k, v

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # split into 3 chunks
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # from hidden_dim (h d) -> heads, dim

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)  # probability mask
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # apply mask to values
        out = rearrange(out, 'b h n d -> b n (h d)')  # rearrange to hidden_dim

        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim,
                              heads=heads,
                              dim_head=dim_head,
                              dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
    ):
        super(ViT, self).__init__()
        """
        Ensure that the image can be cut into equal sections of size (patch_size) i.e.
        image = [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        patch_size = 2

        patch = [
            [1, 1],
            [1, 1],
        ]

        num_patches = (w / patch_size) * (h / patch_size)
                    = (6 / 2) * (6 * 2)
                    = 3 * 3
                    = 9

        Hence, width and height of image must both be multiples of patch_size, 
        preferably a perfect square to ease implementation
        """
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (
            image_width // patch_width
        )  # example (6 // 2) * (6 // 2) = 3 *3 = 9
        patch_dim = channels * patch_height * patch_width  # width * height * channel (flattened dimension assuming square image)

        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # from a combined image (1, 3, 6, 6) -> (1, 9, 4) to 9 flattened 2x2 patches
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(
            1, num_patches + 1, dim))  # 9 patch embeddings + cls_tokens

        self.cls_token = nn.Parameter(torch.randn(
            1, 1, dim))  # random normal tokens ?

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )  # (depth) numbers of (attn + ff)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout+0.2),
            nn.Linear(mlp_dim, num_classes),
        )

        # self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.to_patch_embedding(x)  # embed 2D image into 1D vector
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',
                            b=b)  # create cls tokens dynamically
        x = torch.cat((cls_tokens, x), dim=1)  # append to the 9 embeddings
        x += self.pos_embedding[:, :(
            n + 1)]  # add position embedding of all previous patches up to n+1
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)


# base: 91.37993631999852
# compiled: 90.84370925999974

# if __name__ == '__main__':
#     import timeit

#     torch.backends.cudnn.benchmark = True

#     model = ViT(
#         channels=1,
#         depth=6,
#         dim=64,
#         dim_head=64,
#         dropout=0.1,
#         emb_dropout=0.1,
#         heads=8,
#         image_size=28,
#         mlp_dim=128,
#         num_classes=47,
#         patch_size=7,
#         pool='cls',
#     ).cuda()

#     torch.compile(model, mode='max-autotune')

#     sample = torch.randn((2, 1, 28, 28)).cuda()

#     times = []
#     for i in range(10):
#         times.append(
#             timeit.timeit("model(sample)", globals=globals(), number=25000))

#     print('TIME:', np.mean(times))
