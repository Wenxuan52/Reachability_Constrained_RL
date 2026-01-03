"""Diffusion utilities for Safe Score Matching agent."""
from functools import partial
from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> jnp.ndarray:
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> jnp.ndarray:
    return jnp.linspace(beta_start, beta_end, timesteps)


def vp_beta_schedule(timesteps: int) -> jnp.ndarray:
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,
        a: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)
        return self.reverse_encoder_cls()(reverse_input, training=training)


@partial(
    jax.jit,
    static_argnames=("actor_apply_fn", "act_dim", "T", "clip_sampler", "training"),
)
def ddpm_sampler(
    actor_apply_fn,
    actor_params,
    T: int,
    rng: jax.random.PRNGKey,
    act_dim: int,
    observations: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_hats: jnp.ndarray,
    betas: jnp.ndarray,
    sample_temperature: float,
    clip_sampler: bool,
    training: bool = False,
):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng_inner = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn(
            {"params": actor_params},
            observations,
            current_x,
            input_time,
            training=training,
        )

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = (1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time]))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng_inner, key = jax.random.split(rng_inner, 2)
        z = jax.random.normal(key, shape=(observations.shape[0], current_x.shape[1]))
        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)
        current_x = jnp.clip(current_x, -1, 1) if clip_sampler else current_x
        return (current_x, rng_inner), ()

    key, rng = jax.random.split(rng, 2)
    (action_0, rng), () = jax.lax.scan(
        fn,
        (jax.random.normal(key, (batch_size, act_dim)), rng),
        jnp.arange(T - 1, -1, -1),
        unroll=5,
    )
    action_0 = jnp.clip(action_0, -1, 1)
    return action_0, rng


@partial(
    jax.jit,
    static_argnames=("actor_apply_fn", "act_dim", "T", "clip_sampler", "training"),
)
def ddpm_sampler_keepinner(
    actor_apply_fn,
    actor_params,
    T: int,
    rng: jax.random.PRNGKey,
    act_dim: int,
    observations: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_hats: jnp.ndarray,
    betas: jnp.ndarray,
    sample_temperature: float,
    clip_sampler: bool,
    training: bool = False,
):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, logprob_total, rng_inner = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn(
            {"params": actor_params},
            observations,
            current_x,
            input_time,
            training=training,
        )

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = (1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time]))
        current_x_plus = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng_inner, key = jax.random.split(rng_inner, 2)
        z = jax.random.normal(key, shape=(observations.shape[0], current_x.shape[1]))
        z_scaled = sample_temperature * z
        current_z = current_x_plus + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        logprobs = -0.5 * jnp.sum((current_z - current_x) ** 2, axis=-1)
        condition = (time > 0) & (jnp.sqrt(betas[time]) * sample_temperature > 0)
        logprobs = jax.lax.cond(
            condition,
            lambda _: (1 / (betas[time] * sample_temperature**2)) * logprobs,
            lambda _: jnp.zeros_like(logprobs),
            operand=None,
        )
        logprob_total = logprob_total + logprobs

        current_x = jnp.clip(current_z, -1, 1) if clip_sampler else current_x
        return (current_x, logprob_total, rng_inner), ()

    key, rng = jax.random.split(rng, 2)
    (action_0, logprob_total, rng), () = jax.lax.scan(
        fn,
        (jax.random.normal(key, (batch_size, act_dim)), jnp.zeros(batch_size), rng),
        jnp.arange(T - 1, -1, -1),
        unroll=5,
    )
    action_0 = jnp.clip(action_0, -1, 1)
    logprob_total = (1 / T) * logprob_total
    return action_0, logprob_total, rng
