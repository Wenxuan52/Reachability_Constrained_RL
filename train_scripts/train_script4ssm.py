#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
import yaml

from buffer import ReplayBufferWithCost
from learners.ssm.safe_matching_learner import SafeScoreMatchingLearner
from learners.ssm.safe_matching_config import get_config as get_ssm_config
from safe_control_gym.utils.registration import make


def make_env(seed: int, quad_cfg: Dict):
    env = make("quadrotor", **quad_cfg)
    env.seed(seed)
    return env


def setup_tf_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def build_log_dir(args):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base = os.path.join(
        "results",
        "Safexp-QuadrotorTrajectoryTracking-v0",
        "quadrotor_ssm",
        f"{time_now}_seed{args.seed}",
    )
    os.makedirs(base, exist_ok=True)
    return base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_training", type=int, default=10000)
    parser.add_argument("--buffer_size", type=int, default=500000)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--test_iter_list", type=int, nargs="*", default=None)
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--fixed_steps", type=int, default=None)
    parser.add_argument("--config", type=str, default="train_scripts/env_configs/ssm_constrained_tracking.yaml")
    parser.add_argument("--ddpm_temperature", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--clip_sampler", action="store_true")
    parser.add_argument("--beta_schedule", type=str, default="vp")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--safety_lr", type=float, default=3e-4)
    parser.add_argument("--safety_threshold", type=float, default=0.0)
    parser.add_argument("--cost_limit", type=float, default=10.0)
    parser.add_argument("--safety_discount", type=float, default=0.99)
    parser.add_argument("--safety_lambda", type=float, default=1.0)
    parser.add_argument("--alpha_coef", type=float, default=0.1)
    parser.add_argument("--M_q", type=float, default=1.0)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()
    if args.random_seed is None:
        args.random_seed = args.seed
    return args


def load_quad_cfg(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["quadrotor_config"]


def init_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(logdir=log_dir)


def to_tf_batch(batch):
    obs, act, rew, next_obs, done, cost, _ = batch
    not_terminated = 1.0 - done
    return {
        "observations": tf.convert_to_tensor(obs, dtype=tf.float32),
        "actions": tf.convert_to_tensor(act, dtype=tf.float32),
        "rewards": tf.convert_to_tensor(rew, dtype=tf.float32),
        "next_observations": tf.convert_to_tensor(next_obs, dtype=tf.float32),
        "not_terminated": tf.convert_to_tensor(not_terminated, dtype=tf.float32),
        "costs": tf.convert_to_tensor(cost, dtype=tf.float32),
    }


def evaluate_policy(agent, env, episodes: int, fixed_steps=None):
    returns = []
    violations = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        total_violation = 0.0
        step = 0
        while not done:
            action, _ = agent.act(np.array(obs, dtype=np.float32), deterministic=True)
            action = np.asarray(action)
            next_obs, reward, done, info = env.step(action)
            total_r += reward
            total_violation += float(info.get("constraint_violation", 0.0))
            obs = next_obs
            step += 1
            if fixed_steps is not None and step >= fixed_steps:
                break
        returns.append(total_r)
        violations.append(total_violation)
    return {
        "eval/episode_return": float(np.mean(returns)),
        "eval/episode_return_std": float(np.std(returns)),
        "eval/constraint_violation": float(np.mean(violations)),
        "eval/constraint_violation_std": float(np.std(violations)),
    }


def main():
    args = parse_args()
    quad_cfg = load_quad_cfg(args.config)
    setup_tf_gpu()
    if args.fixed_steps is None:
        args.fixed_steps = int(
            quad_cfg.get("episode_len_sec", 6) * quad_cfg.get("ctrl_freq", 60)
        )
    log_dir = build_log_dir(args) if args.mode == "training" else args.test_dir
    writer = init_writer(log_dir)

    if args.mode == "training":
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = make_env(args.seed, quad_cfg)
    eval_env = make_env(args.seed + 42, quad_cfg)

    obs_space = env.observation_space
    act_space = env.action_space

    buffer = ReplayBufferWithCost(args, buffer_id=0)
    buffer._maxsize = args.buffer_size
    buffer.replay_batch_size = args.batch_size
    buffer.replay_starts = args.start_training

    if args.mode == "testing":
        assert args.test_dir is not None
        cfg_path = os.path.join(args.test_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                saved_args = json.load(f)
            args.T = saved_args.get("T", args.T)
            args.clip_sampler = saved_args.get("clip_sampler", args.clip_sampler)
            args.beta_schedule = saved_args.get("beta_schedule", args.beta_schedule)
            args.ddpm_temperature = saved_args.get("ddpm_temperature", args.ddpm_temperature)
            args.cost_limit = saved_args.get("cost_limit", args.cost_limit)
            args.safety_discount = saved_args.get("safety_discount", args.safety_discount)
            args.safety_lambda = saved_args.get("safety_lambda", args.safety_lambda)
            args.alpha_coef = saved_args.get("alpha_coef", args.alpha_coef)
            args.M_q = saved_args.get("M_q", args.M_q)
            args.safety_threshold = saved_args.get("safety_threshold", args.safety_threshold)

        ssm_cfg = get_ssm_config()
        learner = SafeScoreMatchingLearner.create(
            seed=args.seed,
            observation_space=obs_space,
            action_space=act_space,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            safety_lr=args.safety_lr,
            T=args.T,
            clip_sampler=args.clip_sampler,
            beta_schedule=args.beta_schedule,
            ddpm_temperature=args.ddpm_temperature,
            cost_limit=args.cost_limit,
            safety_discount=args.safety_discount,
            safety_lambda=args.safety_lambda,
            alpha_coef=args.alpha_coef,
            M_q=args.M_q,
            safety_threshold=args.safety_threshold,
            actor_hidden_dims=tuple(ssm_cfg.actor_hidden_dims),
            critic_hidden_dims=tuple(ssm_cfg.critic_hidden_dims),
            safety_hidden_dims=tuple(ssm_cfg.safety_hidden_dims),
        )

        ckpt_dir = os.path.join(args.test_dir, "checkpoints") if args.test_dir else None
        SafeScoreMatchingLearner.load(learner, ckpt_dir or args.test_dir)
        results = evaluate_policy(learner, eval_env, args.num_eval_episode, args.fixed_steps)
        print(json.dumps(results, indent=2))
        with open(os.path.join(args.test_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    ssm_cfg = get_ssm_config()
    learner = SafeScoreMatchingLearner.create(
        seed=args.seed,
        observation_space=obs_space,
        action_space=act_space,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        safety_lr=args.safety_lr,
        T=args.T,
        clip_sampler=args.clip_sampler,
        beta_schedule=args.beta_schedule,
        ddpm_temperature=args.ddpm_temperature,
        cost_limit=args.cost_limit,
        safety_discount=args.safety_discount,
        safety_lambda=args.safety_lambda,
        alpha_coef=args.alpha_coef,
        M_q=args.M_q,
        safety_threshold=args.safety_threshold,
        actor_hidden_dims=tuple(ssm_cfg.actor_hidden_dims),
        critic_hidden_dims=tuple(ssm_cfg.critic_hidden_dims),
        safety_hidden_dims=tuple(ssm_cfg.safety_hidden_dims),
    )

    obs = env.reset()
    episode_return = 0.0
    episode_cost = 0.0
    episode_len = 0

    for step in range(1, args.total_steps + 1):
        if step < args.start_training:
            action = env.action_space.sample()
        else:
            action, _ = learner.act(np.array(obs, dtype=np.float32))
            action = np.asarray(action)

        next_obs, reward, done, info = env.step(action)
        cost = float(info.get("constraint_violation", 0.0))
        buffer.add(obs, action, reward, next_obs, done, cost, None, 1.0)

        obs = next_obs
        episode_return += reward
        episode_cost += cost
        episode_len += 1

        if done:
            writer.add_scalar("train/episode_return", episode_return, step)
            writer.add_scalar("train/constraint_violation", episode_cost, step)
            obs = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            episode_len = 0

        if step >= args.start_training and len(buffer) >= args.start_training:
            batch = buffer.sample(args.batch_size)
            tf_batch = to_tf_batch(batch)
            learner, metrics = learner.update(tf_batch)
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", float(np.array(v)), step)

        if step % args.log_interval == 0:
            writer.flush()

        if step % args.eval_interval == 0:
            eval_metrics = evaluate_policy(learner, eval_env, args.num_eval_episode, args.fixed_steps)
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, step)
            writer.flush()

        if args.save_model and step % args.save_interval == 0:
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            learner.save(ckpt_dir, step)

    writer.close()


if __name__ == "__main__":
    main()
