"""Minimal smoke test for SafeScoreMatchingLearner."""
import os
import shutil

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from learners.ssm import SafeScoreMatchingLearner, prepare_batch


def main():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    learner = SafeScoreMatchingLearner.create(
        seed=0,
        observation_space=obs_space,
        action_space=act_space,
        T=4,
    )

    batch_size = 8
    dummy_batch = {
        "observations": np.random.randn(batch_size, *obs_space.shape).astype(np.float32),
        "actions": np.random.uniform(-1, 1, size=(batch_size, act_space.shape[0])).astype(np.float32),
        "next_observations": np.random.randn(batch_size, *obs_space.shape).astype(np.float32),
        "rewards": np.random.randn(batch_size).astype(np.float32),
        "not_terminated": np.ones(batch_size, dtype=np.float32),
        "costs": np.zeros(batch_size, dtype=np.float32),
    }
    batch = prepare_batch(dummy_batch)

    learner, metrics = learner.update(batch)
    print("Update metrics keys:", list(metrics.keys())[:5])

    action, learner, _ = learner.act(obs_space.sample())
    print("Sampled action shape:", action.shape)

    ckpt_dir = "./temp_ssm_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    learner.save(ckpt_dir)
    restored = learner.load(ckpt_dir)
    print("Restored learner type:", type(restored))
    shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    main()
