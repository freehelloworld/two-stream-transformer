import torch
import torch.nn as nn

SEQ_SIZE = 9


class SequenceEmbed(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32768, 4096)
        self.fc2 = nn.Linear(4096, 200)
        self.relu = nn.ReLU()

    """ Transform x1 into shape (batch_size, 9, 200), x2 into shape (batch_size, 9, 200)

    Parameters:
        x1 : c3d feature with shape (batch_size, 9, 32768)
        x2 : position feature with shape (batch_size, 9, 100, 2)
    """

    def forward(self, x1, x2):
        B = x2.size(0)
        x1 = self.relu(self.fc1(x1))
        x1 = self.relu(self.fc2(x1))
        x2 = x2.reshape(B, SEQ_SIZE, -1)
        return x1, x2


class Attention(nn.Module):

    def __init__(self, dim, n_heads=10, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        # print('attention x:', x.shape)

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        # print('qkv:', qkv.shape)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(
            x
        )  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )  # (n_samples, n_patches + 1, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

        self.fc = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        x1 = x1 + self.attn(self.norm1(x1))
        x2 = x2 + self.attn(self.norm1(x2))
        x = torch.cat((x1, x2), dim=2)

        x = self.act(self.fc(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            n_classes=2,
            sequence_size=SEQ_SIZE,
            embed_dim=200,
            depth=4,
            n_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.seq_embed = SequenceEmbed()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + sequence_size, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x1, x2):
        x1, x2 = self.seq_embed(x1, x2)

        batch_size = x1.shape[0]

        cls_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, embed_dim)

        # print('cls_token:', cls_token.shape)

        x1 = torch.cat((cls_token, x1), dim=1)  # (batch_size, 1 + sequence_size, embed_dim)

        # print('cat cls_token with x:', x.shape)

        x1 = x1 + self.pos_embed  # (batch_size, 1 + sequence_size, embed_dim)

        x1 = self.pos_drop(x1)

        x2 = torch.cat((cls_token, x2), dim=1)  # (batch_size, 1 + sequence_size, embed_dim)

        # print('cat cls_token with x:', x.shape)

        x2 = x2 + self.pos_embed  # (batch_size, 1 + sequence_size, embed_dim)

        # print('prepend x to position embed:', x.shape)

        x2 = self.pos_drop(x2)

        for block in self.blocks:
            x = block(x1, x2)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x
