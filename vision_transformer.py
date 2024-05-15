import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used in the Vision Transformer architecture.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to in_features if not provided.
        out_features (int, optional): Number of output features. Defaults to in_features if not provided.
        act_layer (nn.Module, optional): Activation layer. Defaults to GELU.
        drop (float, optional): Dropout rate. Defaults to 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # Set output features to input features if not specified
        hidden_features = hidden_features or in_features  # Set hidden features to input features if not specified
        self.fc1 = nn.Linear(in_features, hidden_features)  # First fully connected layer
        self.act = act_layer()  # Activation layer (GELU by default)
        self.fc2 = nn.Linear(hidden_features, out_features)  # Second fully connected layer
        self.drop = nn.Dropout(drop)  # Dropout layer

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        
        # Example usage
        # mlp = Mlp(in_features=768, hidden_features=3072, out_features=768, drop=0.1)
        # x = torch.randn(16, 768)  # Example input tensor with batch size 16 and 768 features
        # output = mlp(x)  # Forward pass
        # print(output.shape)  # Should print torch.Size([16, 768])
        """
        x = self.fc1(x)  # Linear transformation, shape: (batch_size, hidden_features)
        x = self.act(x)  # Apply activation function, shape remains: (batch_size, hidden_features)
        x = self.drop(x)  # Apply dropout, shape remains: (batch_size, hidden_features)
        x = self.fc2(x)  # Linear transformation, shape: (batch_size, out_features)
        x = self.drop(x)  # Apply dropout, shape remains: (batch_size, out_features)
        return x  # Return the output tensor



class Attention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) mechanism used in the Vision Transformer architecture.

    Args:
        dim (int): Input dimension (number of input features).
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, add a learnable bias to the query, key, and value projections. Defaults to False.
        qk_scale (float, optional): Override default scale factor for attention scores. Defaults to None.
        attn_drop (float, optional): Dropout rate on attention weights. Defaults to 0.
        proj_drop (float, optional): Dropout rate on output projection. Defaults to 0.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # Dimension per head
        self.scale = qk_scale or head_dim ** -0.5  # Scaling factor for attention scores

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Linear layer to compute query, key, and value
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout layer for attention weights
        self.proj = nn.Linear(dim, dim)  # Linear layer for output projection
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout layer for output projection

    def forward(self, x):
        """
        Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
            torch.Tensor: Attention weights of shape (batch_size, num_heads, num_patches, num_patches).

        # Example usage
        # attn = Attention(dim=768, num_heads=12, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        # x = torch.randn(16, 197, 768)  # Example input tensor with batch size 16, 197 patches, and 768 features
        # output, attn_weights = attn(x)  # Forward pass
        # print(output.shape)  # Should print torch.Size([16, 197, 768])
        # print(attn_weights.shape)  # Should print torch.Size([16, 12, 197, 197])
        """
        B, N, C = x.shape  # Batch size, number of patches, embedding dimension

        # Compute query, key, and value
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Shape of qkv: (3, batch_size, num_heads, num_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Query, key, and value tensors
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Shape of attn: (batch_size, num_heads, num_patches, num_patches)
        attn = attn.softmax(dim=-1)  # Apply softmax to get attention weights
        attn = self.attn_drop(attn)  # Apply dropout to attention weights

        # Compute the output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Shape of x: (batch_size, num_patches, embed_dim)
        x = self.proj(x)  # Apply the output projection
        x = self.proj_drop(x)  # Apply dropout to the output projection
        return x, attn  # Return the output tensor and attention weights




import torch
import torch.nn as nn

class Block(nn.Module):
    """
    Transformer block used in the Vision Transformer (ViT) architecture.

    Args:
        dim (int): Input dimension (number of input features).
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.
        qkv_bias (bool, optional): If True, add a learnable bias to the query, key, and value projections. Defaults to False.
        qk_scale (float, optional): Override default scale factor for attention scores. Defaults to None.
        drop (float, optional): Dropout rate for the output projection. Defaults to 0.
        attn_drop (float, optional): Dropout rate on attention weights. Defaults to 0.
        drop_path (float, optional): Stochastic depth rate. Defaults to 0.
        act_layer (nn.Module, optional): Activation layer. Defaults to GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to LayerNorm.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)  # Layer normalization before attention
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)  # Multi-head attention layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # Stochastic depth (drop path) or identity layer
        self.norm2 = norm_layer(dim)  # Layer normalization before MLP
        mlp_hidden_dim = int(dim * mlp_ratio)  # Hidden dimension for the MLP
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # MLP layer

    def forward(self, x, return_attention=False):
        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, dim).
            return_attention (bool, optional): If True, return the attention weights. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, dim).
            torch.Tensor (optional): Attention weights if return_attention is True.

        # Example usage
        # block = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1)
        # x = torch.randn(16, 197, 768)  # Example input tensor with batch size 16, 197 patches, and 768 features
        # output = block(x)  # Forward pass
        # print(output.shape)  # Should print torch.Size([16, 197, 768])
        """
        # Normalize the input tensor
        y, attn = self.attn(self.norm1(x))  # Apply normalization and attention
        if return_attention:
            return attn  # If requested, return attention weights

        # Add residual connection and apply stochastic depth (drop path)
        # It is a regularization technique
        x = x + self.drop_path(y)  # Residual connection after attention

        # Normalize the tensor and apply MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # Residual connection after MLP

        return x  # Return the output tensor



    
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    Uses a 2D convolutional layer to convert an image into a sequence of patches.
    
    Args:
        img_size (int, optional): The size of the input image (default: 224).
        patch_size (int, optional): The size of each patch (default: 16).
        in_chans (int, optional): The number of input channels (default: 3).
        embed_dim (int, optional): The dimensionality of the embedded patches (default: 768).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # Calculate the number of patches
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size  # Input image size
        self.patch_size = patch_size  # Patch size
        self.num_patches = num_patches  # Total number of patches

        # Convolutional layer to generate patch embeddings
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_chans, img_size, img_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
            # Example usage
            # patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
            # x = torch.randn(16, 3, 224, 224)  # Example input tensor with batch size 16, 3 channels, and image size 224x224
            # output = patch_embed(x)  # Forward pass
            # print(output.shape)  # Should print torch.Size([16, 196, 768])
        """
        B, C, H, W = x.shape  # Batch size, number of channels, height, width

        # Apply the convolutional layer to generate patch embeddings
        x = self.proj(x)  # Shape: (batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt)

        # Flatten the patch embeddings and transpose dimensions to match the expected output shape
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

        return x  # Return the patch embeddings





import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # Number of features (embedding dimension)

        # Patch embedding layer: transforms the image into a sequence of patches
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # Total number of patches

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Learnable class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # Positional embedding for patches + class token
        self.pos_drop = nn.Dropout(p=drop_rate)  # Dropout for positional embeddings

        # Stochastic depth decay rule: different drop rates for each block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])  # List of transformer blocks
        
        self.norm = norm_layer(embed_dim)  # Final normalization layer

        # Classifier head
        self.head = nn.Sequential(*[
            nn.Linear(2 * embed_dim, embed_dim),  # First linear layer
            nn.GELU(),  # GELU activation
            nn.Linear(embed_dim, num_classes)  # Second linear layer
        ]) if num_classes > 0 else nn.Identity()  # Identity layer if num_classes == 0

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)  # Initialize positional embeddings
        trunc_normal_(self.cls_token, std=.02)  # Initialize class token
        self.apply(self._init_weights)  # Initialize all model weights

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # Initialize linear layer weights
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize linear layer biases to 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # Initialize layer norm biases to 0
            nn.init.constant_(m.weight, 1.0)  # Initialize layer norm weights to 1

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1  # Number of patches in the input
        N = self.pos_embed.shape[1] - 1  # Number of patches in the positional embedding
        if npatch == N and w == h:
            return self.pos_embed  # Return positional embedding if size matches
        class_pos_embed = self.pos_embed[:, 0]  # Class token positional embedding
        patch_pos_embed = self.pos_embed[:, 1:]  # Patch tokens positional embedding
        dim = x.shape[-1]  # Embedding dimension
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1  # Small adjustment to avoid floating point error
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',  # Bicubic interpolation
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)  # Reshape back to original form
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)  # Concatenate class token

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape  # Batch size, number of channels, width, height
        x = self.patch_embed(x)  # Apply patch embedding

        # Add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand class token to batch size
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate class token with patch tokens

        # Add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)  # Apply dropout to positional embeddings
    
    def forward(self, x, classify=False):
        '''
        Forward pass for the Vision Transformer.
        Changes:
        1. In the original DINO paper, the authors return the [CLS] token `x[:, 0]`.
        2. We return `x` which contains image information as well; this can be used for reconstruction.

        # Example usage
        # vit = VisionTransformer(img_size=[224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12)
        # x = torch.randn(16, 3, 224, 224)  # Example input tensor with batch size 16 and image size 224x224
        # output = vit(x, classify=True)  # Forward pass for classification
        # print(output.shape)  # Should print torch.Size([16, 1000])

        if classify not true
        # Shape: (batch_size, num_patches + 1, embed_dim)
        '''
        x = self.prepare_tokens(x)  # Prepare tokens (patch + positional embedding)

        # Pass through transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # Apply each transformer block
           
        x = self.norm(x)  # Apply final normalization

        # If classification, return class token and mean of the patch tokens concatenated
        if classify: 
             # Should print torch.Size([16, 1000])
            return self.head(torch.cat((x[:, 0], torch.mean(x[:, 1:], dim=1)), dim=1))
        
        return x  ## Shape: (batch_size, num_patches + 1, embed_dim)
    
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)  # Prepare tokens (patch + positional embedding)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)  # Apply each transformer block
            else:
                # Return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)  # Prepare tokens (patch + positional embedding)
        # Return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # Apply each transformer block
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))  # Apply normalization and collect output
        return output  # Return the list of outputs



def vit_tiny(patch_size=16, **kwargs):
    '''
    Creates a tiny Vision Transformer model with specific parameters.
    
    Args:
        patch_size (int, optional): Size of each patch. Default is 16.
        **kwargs: Additional keyword arguments to pass to the VisionTransformer constructor.
    
    Returns:
        VisionTransformer: A VisionTransformer model instance with a small configuration.
    
    Notes:
        - This configuration uses 12 transformer blocks (depth=12).
        - The embedding dimension is set to 6 (embed_dim=6).
        - The number of attention heads is set to 3 (num_heads=3).
        - The MLP ratio, which determines the hidden layer size in the MLP relative to the embedding dimension, is set to 4 (mlp_ratio=4).
        - Bias terms are used in the QKV projections (qkv_bias=True).
        - The normalization layer uses LayerNorm with an epsilon value of 1e-6.
    '''
    model = VisionTransformer(
        patch_size=patch_size,  # Size of each patch (default 16)
        embed_dim=6,  # Embedding dimension (small value for tiny model)
        depth=12,  # Number of transformer blocks
        num_heads=3,  # Number of attention heads
        mlp_ratio=4,  # Ratio of hidden layer size to embedding dimension in MLP
        qkv_bias=True,  # Use bias terms in QKV projections
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Normalization layer with specific epsilon value
        **kwargs  # Additional keyword arguments
    )
    return model 


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# It is from Author of MCSSL (We are not using this projection head)
class PROJHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        # normalize the weights
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        
        self.data_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.data_layer.weight_g.data.fill_(1)
        self.data_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        
        # project into unit sphere
        x = nn.functional.normalize(x, dim=-1, p=2)
        x_cls = self.last_layer(x[:, 0])
        x_data = self.data_layer(x[:, 1:])
        return x_cls, x_data


class RECHead(nn.Module):
    def __init__(self, in_dim, in_chans=3, patch_size=16):
        super().__init__()

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        self.convTrans = nn.ConvTranspose2d(in_dim, in_chans, kernel_size=(patch_size, patch_size), 
                                                stride=(patch_size, patch_size))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('Rec IN',x.shape)
        x = self.mlp(x)
        
        x_rec = x.transpose(1, 2)
        out_sz = tuple( (  int(math.sqrt(x_rec.size()[2]))  ,   int(math.sqrt(x_rec.size()[2])) ) )
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))
        # print('Rec Out',x_rec.shape)
                
        return x_rec
