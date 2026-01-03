"""TensorFlow implementation of Safe Score Matching learner."""
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf

from .diffusion import DDPM, cosine_beta_schedule, ddpm_sampler, vp_beta_schedule, FourierFeatures
from .mlp import MLP
from .state_action_value import StateActionValue


def mish(x):
    return x * tf.math.tanh(tf.nn.softplus(x))


def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau: float):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign(tau * source_var + (1.0 - tau) * target_var)


class SafeScoreMatchingLearner:
    def __init__(
        self,
        score_model: DDPM,
        critic_1: StateActionValue,
        critic_2: StateActionValue,
        target_critic_1: StateActionValue,
        target_critic_2: StateActionValue,
        safety_critic: StateActionValue,
        target_safety_critic: StateActionValue,
        score_opt: tf.keras.optimizers.Optimizer,
        critic_opt: tf.keras.optimizers.Optimizer,
        safety_opt: tf.keras.optimizers.Optimizer,
        discount: float,
        tau: float,
        safety_tau: float,
        act_dim: int,
        T: int,
        clip_sampler: bool,
        ddpm_temperature: float,
        betas: tf.Tensor,
        alphas: tf.Tensor,
        alpha_hats: tf.Tensor,
        M_q: float,
        cost_limit: float,
        safety_discount: float,
        safety_lambda: float,
        alpha_coef: float,
        safety_threshold: float,
        safety_grad_scale: float,
        safe_lagrange_coef: float,
        seed: int,
    ):
        self.score_model = score_model
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2
        self.safety_critic = safety_critic
        self.target_safety_critic = target_safety_critic
        self.score_opt = score_opt
        self.critic_opt = critic_opt
        self.safety_opt = safety_opt
        self.discount = discount
        self.tau = tau
        self.safety_tau = safety_tau
        self.act_dim = act_dim
        self.T = T
        self.clip_sampler = clip_sampler
        self.ddpm_temperature = ddpm_temperature
        self.betas = betas
        self.alphas = alphas
        self.alpha_hats = alpha_hats
        self.M_q = M_q
        self.cost_limit = cost_limit
        self.safety_discount = safety_discount
        self.safety_lambda = safety_lambda
        self.alpha_coef = alpha_coef
        self.safety_threshold = safety_threshold
        self.safety_grad_scale = safety_grad_scale
        self.safe_lagrange_coef = safe_lagrange_coef
        self._rng = tf.random.Generator.from_seed(seed)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        safety_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        safety_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        safety_tau: Optional[float] = None,
        ddpm_temperature: float = 1.0,
        actor_layer_norm: bool = False,
        T: int = 5,
        time_dim: int = 64,
        clip_sampler: bool = True,
        beta_schedule: str = "vp",
        M_q: float = 1.0,
        cost_limit: float = 25.0,
        safety_discount: float = 0.99,
        safety_lambda: float = 1.0,
        alpha_coef: float = 0.1,
        safety_threshold: float = 0.0,
        safety_grad_scale: float = 1.0,
        safe_lagrange_coef: float = 0.5,
    ):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        act_dim = action_space.shape[-1]

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(T)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(T)
        else:
            betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_hats = tf.math.cumprod(alphas, axis=0)

        preprocess_time = FourierFeatures(output_size=time_dim, learnable=True)
        cond_encoder = MLP([128, 128], activations=[mish, mish], activate_final=True)
        reverse_encoder = MLP(list(actor_hidden_dims) + [act_dim], activations=[mish for _ in actor_hidden_dims] + [None],
                              activate_final=True, use_layer_norm=actor_layer_norm)
        score_model = DDPM(cond_encoder=cond_encoder, reverse_encoder=reverse_encoder, time_preprocess=preprocess_time)

        dummy_obs = tf.convert_to_tensor(observation_space.sample()[None], dtype=tf.float32)
        dummy_act = tf.convert_to_tensor(action_space.sample()[None], dtype=tf.float32)
        dummy_time = tf.zeros((1, 1), dtype=tf.float32)
        _ = score_model(dummy_obs, dummy_act, dummy_time, training=True)

        critic_1 = StateActionValue(critic_hidden_dims)
        critic_2 = StateActionValue(critic_hidden_dims)
        target_critic_1 = StateActionValue(critic_hidden_dims)
        target_critic_2 = StateActionValue(critic_hidden_dims)
        for net in [critic_1, critic_2, target_critic_1, target_critic_2]:
            _ = net(dummy_obs, dummy_act, training=True)
        target_critic_1.set_weights(critic_1.get_weights())
        target_critic_2.set_weights(critic_2.get_weights())

        safety_critic = StateActionValue(safety_hidden_dims)
        target_safety_critic = StateActionValue(safety_hidden_dims)
        _ = safety_critic(dummy_obs, dummy_act, training=True)
        _ = target_safety_critic(dummy_obs, dummy_act, training=True)
        target_safety_critic.set_weights(safety_critic.get_weights())

        # Use legacy optimizers so we can apply gradients to different
        # variable subsets without pre-building the optimizer variable list.
        score_opt = tf.keras.optimizers.legacy.Adam(actor_lr)
        critic_opt = tf.keras.optimizers.legacy.Adam(critic_lr)
        safety_opt = tf.keras.optimizers.legacy.Adam(safety_lr)

        return cls(
            score_model,
            critic_1,
            critic_2,
            target_critic_1,
            target_critic_2,
            safety_critic,
            target_safety_critic,
            score_opt,
            critic_opt,
            safety_opt,
            discount,
            tau,
            safety_tau if safety_tau is not None else tau,
            act_dim,
            T,
            clip_sampler,
            ddpm_temperature,
            tf.convert_to_tensor(betas, dtype=tf.float32),
            tf.convert_to_tensor(alphas, dtype=tf.float32),
            tf.convert_to_tensor(alpha_hats, dtype=tf.float32),
            M_q,
            cost_limit,
            safety_discount,
            safety_lambda,
            alpha_coef,
            safety_threshold,
            safety_grad_scale,
            safe_lagrange_coef,
            seed,
        )

    def act(self, observation, deterministic: bool = False):
        obs = tf.convert_to_tensor(observation[None], dtype=tf.float32)
        actions = ddpm_sampler(
            self.score_model,
            self.T,
            self.act_dim,
            obs,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.clip_sampler,
            training=False,
        )
        action = actions[0]
        if not deterministic:
            noise = self._rng.normal(shape=action.shape, dtype=tf.float32) * 0.1
            action = tf.clip_by_value(action + noise, -1.0, 1.0)
        return action.numpy(), self

    def _ddpm_next_actions(self, observations):
        return ddpm_sampler(
            self.score_model,
            self.T,
            self.act_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.clip_sampler,
            training=True,
        )

    def update_q(self, batch: Dict[str, tf.Tensor]):
        obs, actions = batch["observations"], batch["actions"]
        next_obs, rewards, not_done = batch["next_observations"], batch["rewards"], batch["not_terminated"]

        next_actions = self._ddpm_next_actions(next_obs)
        next_actions = tf.clip_by_value(next_actions + 0.1 * tf.random.normal(tf.shape(next_actions)), -1.0, 1.0)

        next_q1 = self.target_critic_1(next_obs, next_actions, training=True)
        next_q2 = self.target_critic_2(next_obs, next_actions, training=True)
        next_v = tf.minimum(next_q1, next_q2)
        target_q = rewards + self.discount * not_done * next_v

        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic_1(obs, actions, training=True)
            q2 = self.critic_2(obs, actions, training=True)
            loss1 = tf.reduce_mean(tf.square(q1 - tf.stop_gradient(target_q)))
            loss2 = tf.reduce_mean(tf.square(q2 - tf.stop_gradient(target_q)))

        grads1 = tape.gradient(loss1, self.critic_1.trainable_variables)
        grads2 = tape.gradient(loss2, self.critic_2.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads1, self.critic_1.trainable_variables))
        self.critic_opt.apply_gradients(zip(grads2, self.critic_2.trainable_variables))

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return {
            "critic_loss1": loss1,
            "critic_loss2": loss2,
            "target_q_mean": tf.reduce_mean(target_q),
        }

    def _safety_targets(self, batch: Dict[str, tf.Tensor]):
        obs, actions = batch["observations"], batch["actions"]
        next_obs, not_done, costs = batch["next_observations"], batch["not_terminated"], batch["costs"]
        next_actions = self._ddpm_next_actions(next_obs)
        next_actions = tf.clip_by_value(next_actions + 0.1 * tf.random.normal(tf.shape(next_actions)), -1.0, 1.0)
        next_qh = self.target_safety_critic(next_obs, next_actions, training=True)
        next_vh = tf.maximum(0.0, next_qh)
        current_qh = self.safety_critic(obs, actions, training=True)
        current_vh = tf.maximum(0.0, current_qh)
        alpha_term = self.alpha_coef * current_vh
        candidate = self.safety_discount * not_done * next_vh - current_vh + alpha_term
        positive_candidate = tf.maximum(0.0, candidate)
        stage_violation = tf.maximum(0.0, costs - self.cost_limit)
        target = tf.maximum(stage_violation, positive_candidate)
        return target, current_qh, stage_violation

    def update_safety(self, batch: Dict[str, tf.Tensor]):
        target, current_qh, stage_violation = self._safety_targets(batch)

        with tf.GradientTape() as tape:
            qh_pred = self.safety_critic(batch["observations"], batch["actions"], training=True)
            relu_pred = tf.maximum(0.0, qh_pred)
            diff = relu_pred - tf.stop_gradient(target)
            hinge = tf.maximum(0.0, qh_pred - tf.stop_gradient(target))
            loss = tf.reduce_mean(diff ** 2 + self.safety_lambda * hinge ** 2)

        grads = tape.gradient(loss, self.safety_critic.trainable_variables)
        self.safety_opt.apply_gradients(zip(grads, self.safety_critic.trainable_variables))
        soft_update(self.target_safety_critic, self.safety_critic, self.safety_tau)

        return {
            "safety_loss": loss,
            "qh_target_mean": tf.reduce_mean(target),
            "qh_stage_mean": tf.reduce_mean(stage_violation),
            "qh_pred_mean": tf.reduce_mean(current_qh),
        }

    def _critic_jacobian(self, critic_net: StateActionValue, obs: tf.Tensor, actions: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_vals = critic_net(obs, actions, training=True)
        grad = tape.gradient(tf.reduce_sum(q_vals), actions)
        # Safeguard: if the gradient path is broken (e.g., mixed precision or
        # accidental stop_gradient), fall back to zeros to keep actor updates
        # numerically stable instead of crashing.
        if grad is None:
            grad = tf.zeros_like(actions)
        return grad

    def update_actor(self, batch: Dict[str, tf.Tensor]):
        obs, actions = batch["observations"], batch["actions"]
        B = tf.shape(actions)[0]
        time_indices = tf.random.uniform((B,), minval=0, maxval=self.T, dtype=tf.int32)
        noise_sample = tf.random.normal((B, self.act_dim))
        alpha_hats = tf.gather(self.alpha_hats, time_indices)
        alpha_1 = tf.sqrt(alpha_hats)[:, None]
        alpha_2 = tf.sqrt(1.0 - alpha_hats)[:, None]
        noisy_actions = alpha_1 * actions + alpha_2 * noise_sample
        time_embed = tf.cast(time_indices[:, None], tf.float32)

        critic_j1 = self._critic_jacobian(self.critic_1, obs, noisy_actions)
        critic_j2 = self._critic_jacobian(self.critic_2, obs, noisy_actions)
        critic_jacobian = (critic_j1 + critic_j2) / 2.0

        with tf.GradientTape() as tape:
            tape.watch(noisy_actions)
            safety_q = self.safety_critic(obs, noisy_actions, training=True)
        safety_value = tf.maximum(0.0, safety_q)
        safety_mask = safety_value <= self.safety_threshold
        safety_jacobian = tape.gradient(tf.reduce_sum(safety_q), noisy_actions)
        if safety_jacobian is None:
            safety_jacobian = tf.zeros_like(noisy_actions)

        phi = tf.where(
            safety_mask[:, None],
            self.M_q * critic_jacobian,
            -self.M_q * self.safety_grad_scale * safety_jacobian,
        )

        with tf.GradientTape() as tape:
            eps_pred = self.score_model(obs, noisy_actions, time_embed, training=True)
            target = -phi
            matching_loss = tf.reduce_mean(tf.square(target - eps_pred))

            sampled_actions = self._ddpm_next_actions(obs)
            q1 = self.critic_1(obs, sampled_actions, training=True)
            q2 = self.critic_2(obs, sampled_actions, training=True)
            q_min = tf.minimum(q1, q2)
            qc = self.safety_critic(obs, sampled_actions, training=True)
            penalty = tf.maximum(0.0, qc - self.safety_threshold)
            actor_loss = tf.reduce_mean(-q_min + self.safe_lagrange_coef * penalty)
            total_loss = actor_loss + 0.5 * matching_loss

        grads = tape.gradient(total_loss, self.score_model.trainable_variables)
        self.score_opt.apply_gradients(zip(grads, self.score_model.trainable_variables))

        return {
            "matching_loss": matching_loss,
            "actor_loss": actor_loss,
            "total_actor_loss": total_loss,
            "phi_norm": tf.reduce_mean(tf.norm(phi, axis=-1)),
            "safety_mask_ratio": tf.reduce_mean(tf.cast(safety_mask, tf.float32)),
        }

    def update(self, batch: Dict[str, tf.Tensor]):
        critic_info = self.update_q(batch)
        safety_info = self.update_safety(batch)
        actor_info = self.update_actor(batch)
        metrics = {}
        metrics.update({k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in critic_info.items()})
        metrics.update({k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in safety_info.items()})
        metrics.update({k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in actor_info.items()})
        return self, metrics

    def save(self, ckpt_dir: str, step: int):
        ckpt = tf.train.Checkpoint(
            score_model=self.score_model,
            critic_1=self.critic_1,
            critic_2=self.critic_2,
            target_critic_1=self.target_critic_1,
            target_critic_2=self.target_critic_2,
            safety_critic=self.safety_critic,
            target_safety_critic=self.target_safety_critic,
            score_opt=self.score_opt,
            critic_opt=self.critic_opt,
            safety_opt=self.safety_opt,
        )
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
        manager.save(checkpoint_number=step)

    @classmethod
    def load(cls, learner: "SafeScoreMatchingLearner", ckpt_dir: str):
        ckpt = tf.train.Checkpoint(
            score_model=learner.score_model,
            critic_1=learner.critic_1,
            critic_2=learner.critic_2,
            target_critic_1=learner.target_critic_1,
            target_critic_2=learner.target_critic_2,
            safety_critic=learner.safety_critic,
            target_safety_critic=learner.target_safety_critic,
            score_opt=learner.score_opt,
            critic_opt=learner.critic_opt,
            safety_opt=learner.safety_opt,
        )
        latest = tf.train.latest_checkpoint(ckpt_dir)
        ckpt.restore(latest).expect_partial()
        return learner

    def get_weights(self):
        return {
            "score_model": self.score_model.get_weights(),
            "critic_1": self.critic_1.get_weights(),
            "critic_2": self.critic_2.get_weights(),
            "target_critic_1": self.target_critic_1.get_weights(),
            "target_critic_2": self.target_critic_2.get_weights(),
            "safety_critic": self.safety_critic.get_weights(),
            "target_safety_critic": self.target_safety_critic.get_weights(),
        }

    def set_weights(self, weights):
        self.score_model.set_weights(weights["score_model"])
        self.critic_1.set_weights(weights["critic_1"])
        self.critic_2.set_weights(weights["critic_2"])
        self.target_critic_1.set_weights(weights["target_critic_1"])
        self.target_critic_2.set_weights(weights["target_critic_2"])
        self.safety_critic.set_weights(weights["safety_critic"])
        self.target_safety_critic.set_weights(weights["target_safety_critic"])

