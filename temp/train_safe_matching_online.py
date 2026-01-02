#! /usr/bin/env python
from __future__ import annotations

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import SafeScoreMatchingLearner, TD3Learner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_safety_env
from jaxrl5.evaluation import evaluate
from jaxrl5.utils import append_history
from jaxrl5.wrappers import SafetyRecordEpisodeStatistics, WANDBVideo


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_safe_score_matching", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string(
    "env_name",
    "SafetyCarButton1-v0",
    "Safety-Gymnasium environment name.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Evaluation episodes.")
flags.DEFINE_integer("log_interval", 400, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Evaluation interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 150_000, "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of steps before learning starts."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("save_video", False, "Upload videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("epoch_length", 400, "Number of environment steps per epoch statistic.")
config_flags.DEFINE_config_file(
    "config",
    "examples/states/configs/safe_matching_config.py",
    "Path to the training hyperparameter configuration.",
    lock_config=False,
)


def _maybe_init_wandb():
    if FLAGS.wandb:
        if FLAGS.run_name:
            wandb.init(
                project=FLAGS.project_name,
                name=FLAGS.run_name,
                tags=[FLAGS.run_name],
            )
        else:
            wandb.init(project=FLAGS.project_name)
        wandb.config.update(FLAGS)


def _make_env(env_name: str, seed: int, allow_video: bool = True):
    env = make_safety_env(env_name, seed=seed)
    env = SafetyRecordEpisodeStatistics(env, deque_size=1)
    if allow_video and FLAGS.wandb and FLAGS.save_video:
        env = WANDBVideo(env)
    return env


# def _make_env(env_name: str, seed: int, allow_video: bool = True):
#     env = make_safety_env(env_name, seed=seed)
    
#     try:
#         builder = env.env.env.env
#         task = builder.task
#         if task.world is None:
#             task._build()
#         model = task.world.model
#         old_ts = model.opt.timestep
#         model.opt.timestep = 0.01
#         print(f"[INFO] Changed timestep from {old_ts} → {model.opt.timestep}")
#     except Exception as e:
#         print(f"[WARN] Could not modify timestep automatically: {e}")

#     env = SafetyRecordEpisodeStatistics(env, deque_size=1)
#     if allow_video and FLAGS.wandb and FLAGS.save_video:
#         env = WANDBVideo(env)
#     return env


def _make_eval_policy(agent):
    eval_agent = agent

    def policy(obs):
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(obs)
        return np.asarray(action)

    return policy


def _sample(env, action, episode_length):
    next_obs, reward, cost, terminated, truncated, info = env.step(action)

    if episode_length + 1 >= 400:
        truncated = True

    return next_obs, reward, cost, terminated, truncated, info


def main(_):
    _maybe_init_wandb()

    train_env = _make_env(FLAGS.env_name, seed=FLAGS.seed, allow_video=True)
    eval_env = _make_env(FLAGS.env_name, seed=FLAGS.seed + 42, allow_video=True)

    obs_shape = train_env.observation_space.shape
    act_shape = train_env.action_space.shape

    replay_buffer = ReplayBuffer(obs_shape, act_shape, capacity=FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        train_env.observation_space,
        train_env.action_space,
        **kwargs,
    )

    observation, _ = train_env.reset(seed=FLAGS.seed)
    episode_return, episode_cost, episode_length = 0.0, 0.0, 0
    epoch_reward, epoch_cost = 0.0, 0.0
    experiment_name = FLAGS.run_name or FLAGS.project_name
    latest_cost_mean = None

    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if step < FLAGS.start_training:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            action, agent = agent.sample_actions(observation)
            action = np.asarray(action, dtype=np.float32)

        next_obs, reward, cost, terminated, truncated, info = train_env.step(action)
        # next_obs, reward, cost, terminated, truncated, info = _sample(train_env, action, episode_length)
        done = bool(terminated or truncated)

        replay_buffer.insert(
            observation,
            action,
            float(reward),
            float(cost),
            next_obs,
            terminated,
            truncated,
        )

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1
        epoch_reward += float(reward)
        epoch_cost += float(cost)
        observation = next_obs

        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch)

            if FLAGS.wandb and step % FLAGS.log_interval == 0:
                wandb.log(
                    {f"training/{k}": float(v) for k, v in update_info.items()},
                    step=step,
                )

        if done:
            if FLAGS.wandb:
                wandb.log(
                    {
                        "training/return": episode_return,
                        "training/cost": episode_cost,
                        "training/length": episode_length,
                    },
                    step=step,
                )

            observation, _ = train_env.reset()
            episode_return, episode_cost, episode_length = 0.0, 0.0, 0

        if step % FLAGS.epoch_length == 0:
            epoch_index = step // FLAGS.epoch_length
            latest_cost_mean = epoch_cost / FLAGS.epoch_length
            if FLAGS.wandb:
                wandb.log(
                    {
                        "epoch/index": epoch_index,
                        "epoch/reward_sum": epoch_reward,
                        "epoch/reward_mean": epoch_reward / FLAGS.epoch_length,
                        "epoch/cost_sum": epoch_cost,
                        "epoch/cost_mean": latest_cost_mean,
                        "training/cost_mean": latest_cost_mean,
                    },
                    step=step,
                )
            epoch_reward, epoch_cost = 0.0, 0.0

        if step % FLAGS.eval_interval == 0:
            metrics = evaluate(
                eval_env,
                _make_eval_policy(agent),
                episodes=FLAGS.eval_episodes,
            )
            if FLAGS.wandb:
                wandb.log(metrics, step=step)
            else:
                print(
                    f"[step {step}] return={metrics['eval/return_mean']:.2f} "
                    f"cost={metrics['eval/cost_mean']:.2f} "
                    f"len={metrics['eval/ep_len_mean']:.1f}"
                )

            history_metrics = {
                "eval/return_std": metrics["eval/return_std"],
                "eval/return_mean": metrics["eval/return_mean"],
                "eval/cost_mean": metrics["eval/cost_mean"],
                "eval/cost_std": metrics.get("eval/cost_std", float("nan")),
                # "training/cost_mean": latest_cost_mean
                # if latest_cost_mean is not None
                # else float("nan"),
            }
            append_history(
                step,
                FLAGS.env_name,
                experiment_name,
                FLAGS.seed,
                history_metrics,
            )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    wandb.login(key="7feaca49acb80e68486cc6e9c40b2f2c397a0fae")
    app.run(main)
