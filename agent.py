import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import copy

from models import AtariDQN
from replaybuffer import PrioritizedReplayBuffer
from ewc import EWC


class DQNAgent(nn.Module):
    def __init__(
        self,
        env_list: list[gym.Env],
        num_tasks: int,
        in_channels: int,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        lr: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # other parameters
        mask_loss_weight: float = 0.3,
        ewc_lambda: float = 100
    ):
        super(DQNAgent, self).__init__()

        self.env_list = env_list
        self.env = env_list[0]
        self.num_tasks = num_tasks

        obs_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n
        self.in_channels = in_channels

        self.current_task_id = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.memory_size = memory_size
        self.lr = lr
        self.seed = seed

        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.tau = 0.1 # for polyak averaging
        self.alpha = alpha
        self.beta = beta
        self.beta_initial = beta
        self.prior_eps = prior_eps

        self.mask_loss_weight = mask_loss_weight
        self.ewc_lambda = ewc_lambda

        # replay buffer
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha)

        # dqn
        self.dqn = AtariDQN(in_channels, action_dim, self.num_tasks).to(self.device)
        self.dqn_target = AtariDQN(in_channels, action_dim, self.num_tasks).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # online ewc
        self.ewc = EWC(self, self.ewc_lambda)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        #self.recorder = MinatarVideoRecorder(in_channels, 20)

        self.transition = list()

        self.is_test = False
        self.need_ewc_loss = False

        self.epsilons = []
        self.losses = []
        self.scores = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        # epsilon greedy policy (disabled during test)
        if not self.is_test and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device), self.current_task_id, update_mask_prototypes=True)[0][0].argmax() # important
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, terminated]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples)

        # ewc loss
        if (self.need_ewc_loss == True):
            ewc_loss = self.ewc.ewc_loss()
        else:
            ewc_loss = 0

        loss = torch.mean(elementwise_loss * weights) + ewc_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def train(self, num_frames: int, task_id = 0, plot = False):
        self.is_test = False

        self.env = self.env_list[task_id]
        self.current_task_id = task_id

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        score = 0

        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta_initial + fraction * (1.0 - self.beta_initial)

            if done:
                state, _ = self.env.reset(seed=self.seed)
                self.scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                self.losses.append(loss)
                update_cnt += 1

                self.epsilon = max(
                    self.min_epsilon,
                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                )
                self.epsilons.append(self.epsilon)

                # soft update every step
                #self._target_soft_update(self.tau)

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

        # save fisher at end of task
        fisher = self.ewc.estimate_fisher(batch_size=64, num_batches=200)
        self.ewc.consolidate(fisher)
        self.need_ewc_loss = True
        # clear memory buffer
        self.memory.clear()
        self.epsilon = self.max_epsilon
        self.beta = self.beta_initial

        if plot == True:
            self._plot(frame_idx, self.scores, self.losses, self.epsilons)

        self.losses = []
        self.scores = []
        self.epsilons = []
        
        self.env.close()

    def test(self, video_folder: str, task_id: int, record_video = False) -> float:
        self.is_test = True

        self.env = copy.deepcopy(self.env_list[task_id])
        self.current_task_id = task_id

        self.env = gym.wrappers.RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix=f"task{task_id}",
            episode_trigger=lambda x: True
        )

        self.env.start_recording(f'task{task_id}.mp4')
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0.0
        frame_count = 0
        max_frame_count = 1000

        while not done:
            frame_count += 1
            if frame_count > max_frame_count:
                break

            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            if record_video == True:
                self.recorder.add_frame(next_state)

            state = next_state
            score += reward

        if record_video == True:
            self.recorder.save(f'{video_folder}/recorder_task{task_id}.mp4', fps=5)

        self.env.stop_recording()
        self.env.close()
        self.is_test = False

        return score

    def _compute_dqn_loss(self, samples: dict[str, np.ndarray]) -> torch.Tensor:
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        terminated = torch.FloatTensor(samples["terminated"].reshape(-1, 1)).to(device)

        dqn_cur_state, mask_loss1 = self.dqn(state, self.current_task_id, update_mask_prototypes=True)
        dqn_target_next_state, mask_loss2 = self.dqn_target(next_state, self.current_task_id, update_mask_prototypes=True)
        dqn_next_state, mask_loss3 = self.dqn(next_state, self.current_task_id, update_mask_prototypes=True)

        mask_loss = mask_loss1 + mask_loss2 + mask_loss3

        curr_q_value = dqn_cur_state.gather(1, action)
        next_q_value = (
            dqn_target_next_state
            .gather(  # double DQN
                1, dqn_next_state.argmax(dim=1, keepdim=True)
            )
            .detach()
        )
        mask = 1 - terminated
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        total_loss = elementwise_loss * (1 - self.mask_loss_weight) + mask_loss * self.mask_loss_weight

        return total_loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    """
    def _target_soft_update(self, tau: float):
        sb3_utils.polyak_update(
            params=self.dqn.parameters(),
            target_params=self.dqn_target.parameters(),
            tau=tau
        )
    """

    def set_task_id(self, id):
      self.task_id = id

    def _plot(
        self,
        frame_idx: int,
        scores: list[float],
        losses: list[float],
        epsilons: list[float],
    ):
        plt.close("all")
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-100:])))
        plt.plot(savgol_filter(scores, window_length=21, polyorder=3))

        plt.subplot(132)
        plt.title("loss")
        plt.plot(savgol_filter(losses[1000:], window_length=21, polyorder=3)) # exlude first big drop

        plt.subplot(133)
        plt.title("epsilons")
        plt.plot(epsilons)
