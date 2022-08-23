from typing import Sequence, List, Union, Tuple

import jax
from jax import lax, random, numpy as jnp, jit

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np


class PatchExtractor(nn.Module):
  """Custom module to extract patches from the images.
  
  Args:
    patch_size: Sequence[int]. Image will be divided into patches of the desired size.
    stride: int. Stride length to slide the window for patch extraction.
  
  Raises:
    AssertionError: If `patch_size` is not a sequence with length = 2.
  """
  patch_size: Sequence[int]
  stride: int

  def setup(self):
    assert len(self.patch_size) == 2, "length of `patch_size` should be equal to 2."


  @nn.compact
  def __call__(self, images):
    patches = jax.lax.conv_general_dilated_patches(images[:, None, None, :], 
                                                   (1, self.patch_size[0], self.patch_size[1], 1), 
                                                   (1, self.stride, self.stride, 1), 
                                                   padding="VALID")
    n_patches = (images.shape[1] // self.patch_size[0]) * (images.shape[2] // self.patch_size[1])
    patch_dims = self.patch_size[0] * self.patch_size[1] * images.shape[3]
    image_patches = patches.reshape(images.shape[0], n_patches, patch_dims)

    return image_patches