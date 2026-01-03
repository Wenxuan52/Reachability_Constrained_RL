import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, hidden_dims, activations=None, activate_final: bool = True, use_layer_norm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = []
        self.use_layer_norm = use_layer_norm
        activations = activations or [tf.nn.relu for _ in hidden_dims]
        for idx, units in enumerate(hidden_dims):
            act_fn = activations[idx] if idx < len(activations) else activations[-1]
            self.hidden_layers.append(
                tf.keras.layers.Dense(units, activation=act_fn)
            )
            if self.use_layer_norm:
                self.hidden_layers.append(tf.keras.layers.LayerNormalization())
        self.activate_final = activate_final

    def call(self, inputs, training: bool = False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training) if hasattr(layer, "__call__") else layer(x)
        if not self.activate_final and hasattr(self.hidden_layers[-1], "activation"):
            # If deactivate final, apply linear activation
            x = tf.identity(x)
        return x
