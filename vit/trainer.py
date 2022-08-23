from typing import Sequence, List, Union, Tuple

import jax
from jax import lax, random, numpy as jnp, jit

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np


class TrainingLoop:
  """OOP wrapper around functional training loop.
  
  Args:
    model: VisionTransformer. Model architecture for Vision Transformer.
    train_gen: DataLoader. Dataloader to feed the training data to the model dynamically.
    seed: int. Seed value for random number generator to ensure reproducibility.
    epochs: int. Maximum number of iteration for training.
    learning_rate: float. Learning rate for the optimizer.
    momentum: float. Momentum for the optimizer.
    val_gen: DataLoader. Default=None. Dataloader to feed the validation data to the model dynamically.
  
  TODO: Make the `apply_model` method jittable.
  """
  model_ = None
  dropout_init_rng = None
  
  def __init__(self, model: VisionTransformer, train_gen: DataLoader, seed: int,
               epochs: int, learning_rate: float, momentum: float, 
               weight_decay: float, val_gen=None,):
    self.model = model
    self.train_gen = train_gen
    self.key = seed
    self.rng = jax.random.PRNGKey(self.key)
    self.main_rng, self.init_rng, TrainingLoop.dropout_init_rng = random.split(self.rng, 3)
    self.epochs = epochs
    self.lr = learning_rate
    self.momentum = momentum
    self.weight_decay = weight_decay

    self.full_batch_size = (self.model.batch_size, self.model.image_size[0],
                            self.model.image_size[1], self.model.image_size[2])
    
    self.init_train_state()
    self.class_arg()

  
  def class_arg(self):
    TrainingLoop.model_ = self.model


  def init_train_state(self):
    """Initializes the model's and optimizer's state
    """
    self.variables = self.model.init({'params': self.init_rng, 'dropout': TrainingLoop.dropout_init_rng}, 
                                     jnp.ones(self.full_batch_size),train=True)['params']

    lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.lr,
            boundaries_and_scales=
                {int(len(self.train_gen)*self.epochs*0.6): 0.1,
                 int(len(self.train_gen)*self.epochs*0.85): 0.1}
        )
    self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr_schedule, weight_decay=self.weight_decay)
        )
    self.train_state = train_state.TrainState.create(apply_fn = self.model.apply, tx=self.optimizer, params=self.variables)

  
  @staticmethod
  @jax.jit
  def apply_model(state: train_state.TrainState, images: jnp.ndarray, labels: jnp.ndarray, train):  
    """Calculates the gradients during backpropogation to adjust model's parameters.
    
    Args:
      state: train_state.TrainState. State of the model's params at a particular time.
      model: VisionTransformer. Model architecture for Vision Transformer.
      images: jnp.ndarray. Input images.
      labels: jnp.ndarray. Labels for input images.
    
    Returns:
      grads: flax.core.frozen_dict.FrozenDict. Gradients from backpropogation to update model's params.
      loss: float. Loss function's output value.
      accuracy: float. Accuracy of the model.
    """
    def loss_fn(params, train):
      """categorical-cross entropy loss function
      
      Args:
        params: . Model's params (weights and biases).
      
      Returns:
        loss: float. Loss function's output value.
        logits: jnp.ndarray. Predictions made by the `model`.
      """
      logits = TrainingLoop.model_.apply({'params': params}, images, train=True, rngs={'dropout': TrainingLoop.dropout_init_rng})
      one_hot = jax.nn.one_hot(labels, 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
      return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, train)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy

  
  @staticmethod
  @jax.jit
  def update_model(state: train_state.TrainState, grads: flax.core.frozen_dict.FrozenDict):
    """Updates model's params using calculated gradients.
    
    Args:
      state: train_state.TrainState. State of the model's params at a particular time.
      grads: flax.core.frozen_dict.FrozenDict. Gradients from backpropogation to update model's params.
    
    Returns:
      state: train_state.TrainState. Updated state.
    """
    return state.apply_gradients(grads=grads)

 
  @staticmethod
  def train_epoch(state: train_state.TrainState, gen: DataLoader, batch_size: int, rng: jnp.ndarray):
    """Trains the model for one epoch with batch mode.
    
    Args:
      state: train_state.TrainState. State of the model's params at a particular time.
      model: VisionTransformer. Model architecture for Vision Transformer.
      gen: DataLoader. Dataloader to feed the training data to the model dynamically.
      batch_size: int. Size of a batch to yield.
      rng: jnp.ndarray. Random number seed to ensure reproducibility.
    
    Returns:
      state: train_state.TrainState. State of the model's params at a particular time.
      train_loss. float. Loss for 1 epoch.
      train_accuracy. float. Accuracy achieved for 1 epoch.
    """
    epoch_loss = []
    epoch_accuracy = []
    for (batch_images, batch_labels) in tqdm(gen, desc='Batch Training', leave=False):
      batch_images = batch_images
      batch_labels = batch_labels
      grads, loss, accuracy = TrainingLoop.apply_model(state, batch_images, batch_labels, True)
      state = TrainingLoop.update_model(state, grads)
      epoch_loss.append(loss)
      epoch_accuracy.append(accuracy)

    
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


  @staticmethod
  def train(obj):
    """Drives the training process using created OOP wrapper.
    
    Args:
      obj: TrainingLoop. Instance of the class `TrainingLoop` which drives the training process of the model.
      
    Returns:
      obj.train_state. Final state of model's params for getting inference.
    """
    prev_train_acc = -1
    CKPT_DIR = 'ckpts'
    for epoch in tqdm(range(1, obj.epochs + 1), desc="Training"):
      obj.train_state, train_loss, train_accuracy = TrainingLoop.train_epoch(obj.train_state,
                                                      obj.train_gen,
                                                      obj.model.batch_size,
                                                      obj.main_rng,
                                                      )
      
      if(train_accuracy > prev_train_acc):
        prev_train_acc = train_accuracy
        checkpoints.save_checkpoint(CKPT_DIR, obj.train_state, step=epoch, keep=obj.epochs)
      
      
      print(f"epoch: {epoch}, train_loss: {train_loss}, train_accuracy: {train_accuracy}")

    return obj.train_state
