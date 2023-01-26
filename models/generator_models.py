import torch
import torch.nn as nn
from models.models import *

class nyu_modelA(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int Tuple
        Enter the Height and Width (it is not a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            filters=64,
            num_res_blocks=1,
            img_size=(240,304),
            patch_size=16,
            in_chans=3,
            out_chans = 1,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.n_patches, embed_dim)
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
        # self.head = nn.Linear(embed_dim, n_classes)

        self.token_fold = nn.Fold(output_size = img_size, kernel_size = patch_size, stride = patch_size)
        
        self.conv_skip_lidar = nn.Sequential(
            nn.Conv2d(out_chans, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.GELU(),
        )
        
        self.conv_skip_rgb = nn.Sequential(
            nn.Conv2d(in_chans, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.GELU(),
        )

        self.conv_fold = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.GELU(),
        )

        self.RRDB = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])

        # Final output block
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.GELU(),
            nn.Conv2d(filters, out_chans, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, rgb, lidar):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = rgb.shape[0]
        x = self.patch_embed(rgb)

        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = self.token_fold(x.transpose(1,2))

        x = self.conv_fold(x) + self.conv_skip_lidar(lidar) + self.conv_skip_rgb(rgb)
        
        x = self.RRDB(x)

        x = self.conv(x)
        
        return x

class nyu_modelB(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int Tuple
        Enter the Height and Width (it is not a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            filters=64,
            num_res_blocks=1,
            img_size=(240,304),
            patch_size=16,
            rgb_chans=3,
            lidar_chans=1,
            out_chans = 1,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed_rgb = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=rgb_chans,
                embed_dim=embed_dim,
        )

        self.patch_embed_lidar = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=lidar_chans,
                embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed_lidar.n_patches + self.patch_embed_rgb.n_patches, embed_dim)
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
        # self.head = nn.Linear(embed_dim, n_classes)

        self.token_fold = nn.Fold(output_size = img_size, kernel_size = patch_size, stride = patch_size)
        
        self.conv_skip_lidar = nn.Sequential(
            nn.Conv2d(lidar_chans, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.conv_fusion_lidar = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.conv_skip_rgb = nn.Sequential(
            nn.Conv2d(rgb_chans, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.conv_fusion_rgb = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.conv_fold_rgb = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.conv_fold_lidar = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.RRDB = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])

        # Final output block
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, out_chans, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, rgb, lidar):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = rgb.shape[0]
        x_rgb = self.patch_embed_rgb(rgb)
        x_lidar = self.patch_embed_lidar(lidar)
        
        x = torch.cat((x_rgb, x_lidar), dim=1)

        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x_rgb = self.token_fold(x[:,:-285,:].transpose(1,2))
        x_lidar = self.token_fold(x[:,285:,:].transpose(1,2)) 

        x_rgb = self.conv_fold_rgb(x_rgb) + self.conv_skip_rgb(rgb)
        x_lidar = self.conv_fold_lidar(x_lidar) + self.conv_skip_lidar(lidar)

        x = self.conv_fusion_rgb(x_rgb) + self.conv_fusion_lidar(x_lidar)
        
        x = self.RRDB(x)

        x = self.conv(x)

        return x