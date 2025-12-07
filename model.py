# written using ChatGPT

import torch
import torch.nn as nn


# ---------- Patch Embedding with Conv2d ----------

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 48,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 48 / 8 = 6
        self.num_patches = self.grid_size * self.grid_size  # 6 * 6 = 36

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_chans, img_size, img_size)
        x = self.proj(x)          # (B, embed_dim, 6, 6)
        x = x.flatten(2)          # (B, embed_dim, 36)
        x = x.transpose(1, 2)     # (B, 36, embed_dim)
        return x


# ---------- Image Embedding: [CLS] + Positional Embeddings ----------

class ImageEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 48,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.embed_dim = embed_dim
        self.num_patches = self.patch_embed.num_patches  # 36 for 48x48 with P=8

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for [CLS] + all patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=dropout)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_chans, img_size, img_size)
        returns: (B, 1 + num_patches, embed_dim)
        """
        B = x.shape[0]

        # Patch embeddings: (B, num_patches, D)
        x = self.patch_embed(x)

        # Expand [CLS] token for batch: (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Prepend CLS: (B, 1 + num_patches, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Dropout
        x = self.pos_drop(x)

        return x


# ---------- CNN Image Embedding: [CLS] + Positional Embeddings ----------

class CNNImageEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 48,
        in_chans: int = 1,
        stem_channels: int = 64,
        embed_dim: int = 192,   # match your bigger ViT
        dropout: float = 0.15,
    ):
        super().__init__()
        self.img_size = img_size

        # CNN stem: 48x48 -> 24x24 -> 12x12
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 48 -> 24

            nn.Conv2d(32, stem_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 24 -> 12
        )

        # After two 2x2 pools, spatial size is img_size // 4
        self.grid_size = img_size // 4           # 48 // 4 = 12
        self.num_patches = self.grid_size ** 2   # 144

        # Project CNN channel dim to Transformer embed dim
        self.proj = nn.Linear(stem_channels, embed_dim)

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=dropout)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_chans, img_size, img_size)
        returns: (B, 1 + num_patches, embed_dim)
        """
        B = x.shape[0]

        # CNN features: (B, C_stem, 12, 12)
        x = self.stem(x)
        B, C, H, W = x.shape  # H=W=12

        # Flatten spatial: (B, H*W, C_stem)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project to embed_dim: (B, H*W, embed_dim)
        x = self.proj(x)

        # CLS token stuff
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+N, D)

        # Add positional embeddings
        x = x + self.pos_embed

        # Dropout
        x = self.pos_drop(x)

        return x


# ---------- MLP Block for Transformer ----------

class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ---------- Single Transformer Block ----------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (B, N, D)
        )

        self.mlp = MLP(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, embed_dim)
        """
        # Self-attention block (pre-norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)

        # MLP block (pre-norm)
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out

        return x


# ---------- Transformer Encoder (stack of blocks) ----------

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


# ---------- Full Vision Transformer for Emotion Classification ----------

class VisionTransformerEmotion(nn.Module):
    def __init__(
        self,
        img_size: int = 48,
        patch_size: int = 8,
        in_chans: int = 1,
        num_classes: int = 7,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Image → tokens (with CLS + pos embed)
        self.embedding = ImageEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Classification head: small MLP + final layer of 7
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_chans, img_size, img_size)  e.g. (B, 1, 48, 48)
        returns: (B, num_classes) logits
        """
        # (B, N+1, D)
        x = self.embedding(x)

        # (B, N+1, D)
        x = self.encoder(x)

        # Take CLS token (position 0)
        cls_repr = x[:, 0, :]  # (B, D)

        # Classification head
        logits = self.mlp_head(cls_repr)  # (B, num_classes)

        return logits


# ---------- CNN + Vision Transformer for Emotion Classification ----------

class VisionTransformerEmotionCNN(nn.Module):
    def __init__(
        self,
        img_size: int = 48,
        in_chans: int = 1,
        num_classes: int = 7,
        stem_channels: int = 64,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 768,
        dropout: float = 0.15,
    ):
        super().__init__()

        # CNN-based embedding instead of PatchEmbedding
        self.embedding = CNNImageEmbedding(
            img_size=img_size,
            in_chans=in_chans,
            stem_channels=stem_channels,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Same TransformerEncoder as before
        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 48, 48) -> (B, 1+144, D)
        x = self.embedding(x)

        # Transformer encoder
        x = self.encoder(x)

        # CLS token
        cls_repr = x[:, 0, :]   # (B, D)

        # Head
        logits = self.mlp_head(cls_repr)
        return logits

