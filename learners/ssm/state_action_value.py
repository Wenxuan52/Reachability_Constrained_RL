import tensorflow as tf
from .mlp import MLP


class StateActionValue(tf.keras.Model):
    def __init__(self, hidden_dims, **kwargs):
        super().__init__(**kwargs)
        self.encoder = MLP(hidden_dims, activate_final=True)
        self.out_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, obs, act, training: bool = False):
        x = tf.concat([obs, act], axis=-1)
        x = self.encoder(x, training=training)
        return tf.squeeze(self.out_layer(x), axis=-1)
