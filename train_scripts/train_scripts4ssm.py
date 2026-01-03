#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ray PABAL-style entry for SafeScoreMatchingLearner on quadrotor tracking.

This script mirrors the structure of ``train_script.py``: argument parsing,
Ray initialization, actor composition (workers/buffers/learners/evaluators),
and a training/testing entrypoint. The learner is swapped with the
SafeScoreMatchingLearner defined under ``learners.ssm`` and TensorFlow is
intentionally avoided.
"""
import argparse
import datetime
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

# JAX/XLA memory knobs must be set before importing jax or ray workers spawn.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
import jax
import ray

import dynamics  # noqa: F401  # imported for registration side effects
import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from learners.ssm.safe_matching_learner import SafeScoreMatchingLearner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RESULT_ROOT = "../results/Safexp-QuadrotorTrajectoryTracking-v0/quadrotor_ssm"
DEFAULT_ENV = "quadrotor"
MAX_EPISODE_LEN = 360


@ray.remote
class SimpleReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.storage: List[Tuple[np.ndarray, ...]] = []
        self.next_idx = 0

    def add_batch(self, batch: List[Tuple[np.ndarray, ...]]):
        for transition in batch:
            if len(self.storage) < self.capacity:
                self.storage.append(transition)
            else:
                self.storage[self.next_idx] = transition
                self.next_idx = (self.next_idx + 1) % self.capacity

    def size(self) -> int:
        return len(self.storage)

    def sample(self):
        assert self.storage, "Replay buffer is empty"
        idxs = np.random.choice(len(self.storage), size=self.batch_size, replace=len(self.storage) < self.batch_size)
        batch = [self.storage[i] for i in idxs]
        obs, act, rew, nxt, done, cost = zip(*batch)
        return dict(
            observations=np.asarray(obs, dtype=np.float32),
            actions=np.asarray(act, dtype=np.float32),
            rewards=np.asarray(rew, dtype=np.float32),
            next_observations=np.asarray(nxt, dtype=np.float32),
            dones=np.asarray(done, dtype=np.float32),
            costs=np.asarray(cost, dtype=np.float32),
        )


@ray.remote
class SSMWorker:
    def __init__(self, args, worker_id: int):
        self.args = args
        self.worker_id = worker_id
        self.rng = np.random.RandomState(args.random_seed + worker_id)
        config = deepcopy(args.config)
        config.quadrotor_config["episode_len_sec"] = MAX_EPISODE_LEN / config.quadrotor_config["ctrl_freq"]
        env = make(DEFAULT_ENV, **config.quadrotor_config)
        self.env = env
        self.obs, self.info = self.env.reset()
        self.done = False
        self.cost_key = "constraint_values"
        self.learner = SafeScoreMatchingLearner.create(
            seed=args.random_seed + worker_id,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            safety_lr=args.safety_lr,
            T=args.T,
            beta_schedule=args.beta_schedule,
            ddpm_temperature=args.ddpm_temperature,
        )

    def set_weights(self, weights):
        self.learner = self.learner.set_weights(weights)

    def sample_once(self):
        action, self.learner, _ = self.learner.act(np.asarray(self.obs), deterministic=False)
        obs_tp1, reward, done, info = self.env.step(action)
        if hasattr(self.env, "t") and self.env.t >= MAX_EPISODE_LEN:
            done = np.array([True] * len(done)) if isinstance(done, np.ndarray) else True
        costs = np.max(info.get(self.cost_key, 0.0)) if isinstance(info, dict) else np.max(info[0].get(self.cost_key, 0.0))
        transition = (self.obs.copy(), action.copy(), np.float32(reward), obs_tp1.copy(), np.float32(done), np.float32(costs))
        self.obs = obs_tp1 if not done else self.env.reset()[0]
        return transition, 1

    def sample_batch(self, batch_size: int):
        batch = []
        count = 0
        for _ in range(batch_size):
            transition, used = self.sample_once()
            batch.append(transition)
            count += used
        return batch, count


@ray.remote
class SSMLearnerActor:
    def __init__(self, args):
        config = deepcopy(args.config)
        config.quadrotor_config["episode_len_sec"] = MAX_EPISODE_LEN / config.quadrotor_config["ctrl_freq"]
        env = make(DEFAULT_ENV, **config.quadrotor_config)
        obs_space = env.observation_space
        act_space = env.action_space
        self.learner = SafeScoreMatchingLearner.create(
            seed=args.random_seed,
            observation_space=obs_space,
            action_space=act_space,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            safety_lr=args.safety_lr,
            T=args.T,
            beta_schedule=args.beta_schedule,
            ddpm_temperature=args.ddpm_temperature,
        )
        self.iteration = 0

    def get_weights(self):
        return self.learner.get_weights()

    def set_weights(self, weights):
        self.learner.set_weights(weights)

    def update(self, batch: Dict[str, np.ndarray]):
        self.learner, metrics = self.learner.update(batch, step=self.iteration)
        self.iteration += 1
        return metrics, self.learner.get_weights()

    def act(self, obs: np.ndarray, deterministic: bool = False, rng=None):
        return self.learner.act(obs, deterministic=deterministic, rng=rng)


@ray.remote
class DummyEvaluator:
    def __init__(self, args):
        self.args = args
        config = deepcopy(args.config_eval)
        config.quadrotor_config["episode_len_sec"] = MAX_EPISODE_LEN / config.quadrotor_config["ctrl_freq"]
        self.env = make(DEFAULT_ENV, **config.quadrotor_config)
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def evaluate(self, iteration: int, num_episodes: int = 4):
        returns = []
        violations = []
        learner = SafeScoreMatchingLearner.create(
            seed=self.args.random_seed,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        learner = learner.set_weights(self.weights)
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            total_rew = 0.0
            total_violation = 0
            t = 0
            while not done and t < MAX_EPISODE_LEN:
                action, learner, _ = learner.act(np.asarray(obs), deterministic=True)
                obs, rew, done, info = self.env.step(action)
                total_rew += float(rew)
                total_violation += int(np.max(info.get("constraint_values", 0.0))) if isinstance(info, dict) else int(np.max(info[0].get("constraint_values", 0.0)))
                t += 1
            returns.append(total_rew)
            violations.append(total_violation / max(1, t))
        return dict(mean_return=float(np.mean(returns)), violation_rate=float(np.mean(violations)), iteration=iteration)


def built_ssm_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='training')
    mode = parser.parse_args().mode

    if mode == 'testing':
        parser.add_argument('--test_dir', type=str, required=True)
        parser.add_argument('--test_iter_list', type=int, nargs='+', default=[200000])
        args = parser.parse_args()
        params = json.loads(open(os.path.join(args.test_dir, 'config.json')).read())
        for key, val in params.items():
            parser.add_argument('-' + key, default=val)
        return parser.parse_args()

    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--safety_lr', type=float, default=3e-4)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--beta_schedule', type=str, default='vp')
    parser.add_argument('--ddpm_temperature', type=float, default=1.0)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_learners', type=int, default=1)
    parser.add_argument('--total_steps', type=int, default=2000000)
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=50000)
    parser.add_argument('--log_interval', type=int, default=100)

    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = f"{RESULT_ROOT}/{time_now}_seed{parser.parse_args().random_seed:04d}"
    parser.add_argument('--result_dir', type=str, default=result_dir)
    parser.add_argument('--log_dir', type=str, default=result_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=result_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--test_iter_list', type=int, nargs='+', default=[200000])
    return parser.parse_args()


def main():
    args = built_ssm_parser()
    logger.info('begin training agents with parameter {}'.format(str(args)))

    # build env configs for workers/evaluators and attach to args
    config_factory = ConfigFactory()
    config_factory.parser.set_defaults(overrides=['./env_configs/constrained_tracking_reset.yaml'])
    args.config = config_factory.merge()
    config_factory_eval = ConfigFactory()
    config_factory_eval.parser.set_defaults(overrides=['./env_configs/constrained_tracking_eval.yaml'])
    args.config_eval = config_factory_eval.merge()
    if args.mode == 'training':
        ray.init(object_store_memory=5 * 1024 * 1024 * 1024)
        os.makedirs(args.result_dir, exist_ok=True)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        buffer = SimpleReplayBuffer.remote(args.max_buffer_size, args.replay_batch_size)
        learners = [SSMLearnerActor.remote(args) for _ in range(args.num_learners)]
        workers = [SSMWorker.remote(args, i + 1) for i in range(args.num_workers)]
        evaluator = DummyEvaluator.remote(args)

        # sync initial weights
        base_weights = ray.get(learners[0].get_weights.remote())
        [w.set_weights.remote(base_weights) for w in workers]
        evaluator.set_weights.remote(base_weights)

        total_steps = 0
        iteration = 0
        while total_steps < args.total_steps:
            sample_tasks = [w.sample_batch.remote(args.batch_size // args.num_workers) for w in workers]
            for task in sample_tasks:
                batch, count = ray.get(task)
                buffer.add_batch.remote(batch)
                total_steps += count

            if ray.get(buffer.size.remote()) >= args.replay_batch_size:
                samples = ray.get(buffer.sample.remote())
                metrics_futures = [l.update.remote(samples) for l in learners]
                metrics_list = ray.get(metrics_futures)
                latest_weights = metrics_list[0][1]
                for w in workers:
                    w.set_weights.remote(latest_weights)
                evaluator.set_weights.remote(latest_weights)
                iteration += 1
                if iteration % args.log_interval == 0:
                    logger.info(f"Iter {iteration} total_steps {total_steps} metrics {metrics_list[0][0]}")
                if iteration % args.eval_interval == 0:
                    eval_stats = ray.get(evaluator.evaluate.remote(iteration))
                    logger.info(f"Eval stats {eval_stats}")
                if iteration % args.save_interval == 0:
                    ckpt_dir = os.path.join(args.model_dir, f"iter_{iteration}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    config = deepcopy(args.config)
                    config.quadrotor_config["episode_len_sec"] = MAX_EPISODE_LEN / config.quadrotor_config["ctrl_freq"]
                    env = make(DEFAULT_ENV, **config.quadrotor_config)
                    saver = SafeScoreMatchingLearner.create(
                        seed=args.random_seed,
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                    )
                    saver = saver.set_weights(latest_weights)
                    saver.save(ckpt_dir, iteration)

    elif args.mode == 'testing':
        assert args.test_dir is not None
        weights_dir = args.test_dir
        with open(os.path.join(weights_dir, 'config.json')) as f:
            saved_args = argparse.Namespace(**json.load(f))
        args.__dict__.update(saved_args.__dict__)
        ray.init(object_store_memory=5 * 1024 * 1024 * 1024)
        evaluator = DummyEvaluator.remote(args)
        for test_iter in args.test_iter_list:
            ckpt_dir = os.path.join(weights_dir, 'models', f'iter_{test_iter}')
            config = deepcopy(args.config_eval)
            config.quadrotor_config["episode_len_sec"] = MAX_EPISODE_LEN / config.quadrotor_config["ctrl_freq"]
            env = make(DEFAULT_ENV, **config.quadrotor_config)
            learner = SafeScoreMatchingLearner.create(
                seed=args.random_seed,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            learner = learner.load(ckpt_dir, step=test_iter)
            evaluator.set_weights.remote(learner.get_weights())
            stats = ray.get(evaluator.evaluate.remote(test_iter))
            logger.info(f"Test iter {test_iter} stats {stats}")


if __name__ == '__main__':
    main()
