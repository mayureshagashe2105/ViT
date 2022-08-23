from typing import Sequence, List, Union, Tuple

import jax
from jax import lax, random, numpy as jnp, jit

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np

from .Modules import MLP, PatchExtractor, SelfAttention



class VisionTransformer(nn.Module):
  """Vision Transformer module
  
  Args:
    patch_size: Sequence[int]. Image will be divided into patches of the desired size.
    stride: int. Stride length to slide the window for patch extraction.
    image_size: Sequence[int]. Format: (H, W, C). Size of 1 image from the batch.
    projection_dims: int. Number of dimensions for internal representation of the model.
    atten_heads: int. Number of self attention heads to be used.
    transformer_layers: int. Number of transformer encoders to be used.
    batch_size: int. Size of a batch to yield.
    num_classes. int. Number of target classes.
  
  Raises:
    AssertionError: If input images are not in the format (N, H, W, C).
  """
  patch_size: Sequence[int]
  stride: int
  image_size: Sequence[int]
  projection_dims: int
  hidden_dims: int
  atten_heads: int
  transformer_layers: int
  batch_size: int
  num_classes: int
  dropout_val: float


  def setup(self):

    self.patchify = PatchExtractor(self.patch_size, self.stride)
    self.patch_dims = self.patch_size[0] * self.patch_size[1] * self.image_size[-1]

    self.tokens = MLP([self.projection_dims], activations=["Linear"])
    
    self.class_token = self.param("class_token", lambda rng, shape: random.normal(rng, shape), (1, 1, self.projection_dims))

    self.num_patches = ((self.image_size[0] - 
                         self.patch_size[0]) // self.stride + 1) * ((self.image_size[1] - self.patch_size[1]) // self.stride + 1)

    self.positional_embeddings = self.param("pos_embed", lambda rng, shape: random.normal(rng, shape), 
                                            (1, 1 + self.num_patches, self.projection_dims))
    
    self.transformer_encoders = [SelfAttention(self.projection_dims, self.hidden_dims, 
                                              self.atten_heads, self.dropout_val) for _ in range(self.transformer_layers)]
    
    self.dropout = nn.Dropout(self.dropout_val)
    self.logits_mlp = MLP([self.num_classes], activations=['softmax'])



  @nn.compact
  def __call__(self, inputs, train):
    assert len(inputs.shape) == 4, f"""ViT encoder expected 4D vector as input in the 
format (N, H, W, C) but got {len(inputs.shape)}D vector instead."""

    image_patches = self.patchify(inputs)
    
    tokens = self.tokens(image_patches)
    cls_token = self.class_token.repeat(inputs.shape[0], axis=0)
    
    tokens = jnp.concatenate([cls_token, tokens], axis=1) 
    tokens += self.positional_embeddings[:, :image_patches.shape[1] + 1]
    tokens = self.dropout(tokens, deterministic=not train)
    
    for encoder in self.transformer_encoders:
      tokens = encoder(tokens, train=train)
    
    out = tokens[:, 0]
    logits = self.logits_mlp(out)

   
    return logits
    