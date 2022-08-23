from typing import Sequence, List, Union, Tuple

import jax
from jax import lax, random, numpy as jnp, jit

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np


class SelfAttention(nn.Module):
  projection_dims : int 
  hidden_dims : int
  atten_heads : int 
  dropout_val : float = 0.0
  
  def setup(self):
    self.MHA = nn.MultiHeadDotProductAttention(num_heads=self.atten_heads)
    self.Linear = [
            nn.Dense(self.hidden_dims),
            nn.gelu,
            nn.Dropout(self.dropout_val),
            nn.Dense(self.projection_dims)
        ]
    self.norm1 = nn.LayerNorm()
    self.norm2 = nn.LayerNorm()
    self.dropout = nn.Dropout(self.dropout_val)
  
  @nn.compact
  def __call__(self, inputs, train=True):
    temp = self.norm1(inputs)
    attn_out = self.MHA(inputs_q=temp, inputs_kv=temp)
    inputs = inputs + self.dropout(attn_out, deterministic=not train)

    linear_out = self.norm2(inputs)
    for l in self.Linear:
        linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
    inputs = inputs + self.dropout(linear_out, deterministic=not train)
    return inputs