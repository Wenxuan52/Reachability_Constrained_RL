import tensorflow as tf


def cosine_beta_schedule(timesteps, s: float = 0.008):
    """Cosine schedule from https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    t = tf.linspace(0.0, float(timesteps), steps) / float(timesteps)
    alphas_cumprod = tf.math.cos((t + s) / (1.0 + s) * tf.constant(tf.constant(3.141592653589793) * 0.5)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0.0, 0.999)


def linear_beta_schedule(timesteps, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return tf.linspace(beta_start, beta_end, timesteps)


def vp_beta_schedule(timesteps):
    t = tf.cast(tf.range(1, timesteps + 1), tf.float32)
    T = float(timesteps)
    b_max = 10.0
    b_min = 0.1
    alpha = tf.exp(-b_min / T - 0.5 * (b_max - b_min) * (2.0 * t - 1.0) / (T ** 2))
    betas = 1.0 - alpha
    return betas


class FourierFeatures(tf.keras.layers.Layer):
    def __init__(self, output_size: int, learnable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.learnable = learnable

    def build(self, input_shape):
        if self.learnable:
            self.kernel = self.add_weight(
                "kernel", shape=(self.output_size // 2, int(input_shape[-1])), initializer=tf.keras.initializers.RandomNormal(stddev=0.2)
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.learnable:
            f = 2.0 * tf.constant(tf.constant(3.141592653589793)) * tf.matmul(inputs, self.kernel, transpose_b=True)
        else:
            half_dim = self.output_size // 2
            f = tf.math.log(10000.0) / (half_dim - 1)
            f = tf.exp(tf.range(half_dim, dtype=tf.float32) * -f)
            f = inputs * f
        return tf.concat([tf.math.cos(f), tf.math.sin(f)], axis=-1)


class DDPM(tf.keras.Model):
    def __init__(self, cond_encoder, reverse_encoder, time_preprocess, **kwargs):
        super().__init__(**kwargs)
        self.cond_encoder = cond_encoder
        self.reverse_encoder = reverse_encoder
        self.time_preprocess = time_preprocess

    def call(self, obs, act, time, training: bool = False):
        t_ff = self.time_preprocess(time)
        cond = self.cond_encoder(t_ff, training=training)
        reverse_input = tf.concat([act, obs, cond], axis=-1)
        return self.reverse_encoder(reverse_input, training=training)


def ddpm_sampler(actor_model: tf.keras.Model, T: int, act_dim: int, observations: tf.Tensor,
                 alphas: tf.Tensor, alpha_hats: tf.Tensor, betas: tf.Tensor,
                 sample_temperature: float, clip_sampler: bool, training: bool = False):
    batch_size = tf.shape(observations)[0]
    current_x = tf.random.normal((batch_size, act_dim))

    for t in range(T - 1, -1, -1):
        time = tf.ones((batch_size, 1), dtype=tf.float32) * float(t)
        eps_pred = actor_model(observations, current_x, time, training=training)
        alpha = alphas[t]
        alpha_hat = alpha_hats[t]
        alpha_1 = 1.0 / tf.sqrt(alpha)
        alpha_2 = (1.0 - alpha) / tf.sqrt(1.0 - alpha_hat)
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)
        noise = tf.random.normal((batch_size, act_dim))
        current_x = current_x + (tf.cast(t > 0, tf.float32) * tf.sqrt(betas[t]) * sample_temperature * noise)
        if clip_sampler:
            current_x = tf.clip_by_value(current_x, -1.0, 1.0)
    return tf.clip_by_value(current_x, -1.0, 1.0)
