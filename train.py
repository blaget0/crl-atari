import gymnasium as gym
import ale_py
import os
import torch

from agent import DQNAgent


def save_checkpoint(
    agent,
    path: str,
    save_memory: bool = False,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "dqn":             agent.dqn.state_dict(),
        "dqn_target":      agent.dqn_target.state_dict(),
        "optimizer":       agent.optimizer.state_dict(),
        "current_task_id": agent.current_task_id,
        "epsilon":         agent.epsilon,
        "beta":            agent.beta,
        "need_ewc_loss":   agent.need_ewc_loss,
        "seed":            agent.seed,
        "losses":          agent.losses,
        "scores":          agent.scores,
        "epsilons":        agent.epsilons,
        "ewc":             agent.ewc.state_dict() if hasattr(agent.ewc, "state_dict") else None,
    }

    if save_memory:
        checkpoint["memory"] = agent.memory.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    agent,
    path: str,
    load_memory: bool = False,
    strict: bool = True,
):
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)

    agent.dqn.load_state_dict(checkpoint["dqn"], strict=strict)
    agent.dqn_target.load_state_dict(checkpoint["dqn_target"], strict=strict)

    if checkpoint.get("optimizer") is not None:
        agent.optimizer.load_state_dict(checkpoint["optimizer"])

    agent.current_task_id = checkpoint.get("current_task_id", -1)
    agent.epsilon         = checkpoint.get("epsilon", agent.max_epsilon)
    agent.beta            = checkpoint.get("beta", agent.beta_initial)
    agent.need_ewc_loss   = checkpoint.get("need_ewc_loss", False)

    agent.losses   = checkpoint.get("losses", [])
    agent.scores   = checkpoint.get("scores", [])
    agent.epsilons = checkpoint.get("epsilons", [])

    if checkpoint.get("ewc") is not None and hasattr(agent.ewc, "load_state_dict"):
        agent.ewc.load_state_dict(checkpoint["ewc"])

    if load_memory and "memory" in checkpoint:
        agent.memory.load_state_dict(checkpoint["memory"])

    return checkpoint

num_tasks = 3
env1_id = "PongNoFrameskip-v4"
env2_id = "BreakoutNoFrameskip-v4"
env3_id = "SpaceInvadersNoFrameskip-v4"
n_envs = 4
seed = 123

env1 = gym.make(env1_id, render_mode="rgb_array")
env1 = gym.wrappers.AtariPreprocessing(env1, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
env1 = gym.wrappers.FrameStackObservation(env1, stack_size=4)
obs = env1.reset()

env2 = gym.make(env2_id, render_mode="rgb_array")
env2 = gym.wrappers.AtariPreprocessing(env2, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
env2 = gym.wrappers.FrameStackObservation(env2, stack_size=4)
obs = env2.reset()

env3 = gym.make(env3_id, render_mode="rgb_array")
env3 = gym.wrappers.AtariPreprocessing(env3, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
env3 = gym.wrappers.FrameStackObservation(env3, stack_size=4)
obs = env3.reset()


num_frames = 1000000
memory_size = 100000
batch_size = 64
target_update = 1000
epsilon_decay = 1 / 200000
in_channels = 4
lr = 3e-4

max_epsilon = 1.0
min_epsilon = 0.1
gamma = 0.99
alpha = 0.2
beta = 0.6
prior_eps = 1e-6
mask_loss_weight = 0.3
ewc_lambda = 100

env_list = [env1, env2, env3]

agent = DQNAgent(env_list, num_tasks, in_channels, memory_size, batch_size, target_update, epsilon_decay, lr, 
                 seed, max_epsilon, min_epsilon, gamma, alpha, beta, prior_eps, mask_loss_weight, ewc_lambda)

ckpt_path = "checkpoints/agent.pt"

#checkpoint = load_checkpoint(agent, ckpt_path, load_memory=False)

agent.train(num_frames, task_id = 0)

save_checkpoint(agent, ckpt_path, save_memory=False)

#agent.train(num_frames, task_id = 1)