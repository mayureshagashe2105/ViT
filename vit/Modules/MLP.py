from typing import Sequence, List, Union, Tuple

import jax
from jax import lax, random, numpy as jnp, jit

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np


class MLP(nn.Module):
  """Multi-layer perceptron dataclass.
  
  Args:
    hidden_layer_nodes: Sequence[int]. Number of nodes in hidden layers.
    activations: Sequence[str]. Activation functions to apply at each layer.
  
  Raises:
    AssertionError: If length of `self.activations` is not same as length of `self.hidden_layer_nodes`.
    ValueError: If any value from `self.activations` is not from allowed activation functions.
  """
  hidden_layer_nodes: Sequence[int]
  activations: Sequence[str]

  def setup(self):
    
    assert len(self.hidden_layer_nodes) == len(self.activations), "Activation function for each layer must be provided."

    self.__whitelist_activations = ['celu', 'elu', 'gelu', 'glu', 'log_sigmoid',
                                    'log_softmax', 'relu', 'sigmoid', 
                                    'soft_sign', 'softmax', 'softplus', 
                                    'swish', 'PRelu', 'Linear']

    self.layers = [(nn.Dense(self.hidden_layer_nodes[n]), self.activations[n]) 
                  for n in range(len(self.hidden_layer_nodes)) 
                  if self.activations[n] in self.__whitelist_activations 
                  ]
    
    if len(self.layers) is not len(self.activations):
      raise ValueError(f'Activation function should be one of the {self.__whitelist_activations}') 

  @nn.compact
  def __call__(self, input):
    for layer, activation in self.layers:
      x = layer(input)
      x = self.apply_activation(x, activation)
      return x
  
  @staticmethod
  def apply_activation(input, activation):
    if activation == 'celu': return nn.celu(input)
    elif activation == 'elu': return nn.elu(input)
    elif activation == 'gelu': return nn.gelu(input)
    elif activation == 'glu': return nn.glu(input)
    elif activation == 'log_sigmoid': return nn.log_sigmoid(input)
    elif activation == 'log_softmax': return nn.log_softmax(input)
    elif activation == 'relu': return nn.relu(input)
    elif activation == 'sigmoid': return nn.sigmoid(input)
    elif activation == 'soft_sign': return nn.soft_sign(input)
    elif activation == 'softmax': return nn.softmax(input)
    elif activation == 'softplus': return nn.softplus(input)
    elif activation == 'swish': return nn.swish(input)
    elif activation == 'PRelu': return nn.PRelu(input)
    elif activation == "Linear": return input