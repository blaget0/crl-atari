import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#import seaborn as sns
#import imageio
#import cv2
#from IPython.display import Video

from mmu import MaskedMemoryUnit


class AtariDQN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, num_tasks: int):
        super(AtariDQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,  out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.mmu = MaskedMemoryUnit(
            in_features=64 * 7 * 7,
            layer_size=512,
            mask_size=128,
            task_emb_size=32,
            num_tasks=num_tasks,
            out_features=512)

        self.v_head   = nn.utils.spectral_norm(nn.Linear(in_features=512, out_features=1))
        self.adv_head = nn.utils.spectral_norm(nn.Linear(in_features=512, out_features=num_actions))
        self.relu = nn.ReLU()

    def forward(self, x, task_id, update_mask_prototypes=False):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mmu_output, mask_loss, scores = self.mmu(x, task_id)

        v   = self.v_head(mmu_output)
        adv = self.adv_head(mmu_output)
        q = v + (adv - adv.mean(dim=1, keepdim=True))

        if update_mask_prototypes == True:
          self._update_mask_prototypes(task_id, scores)

        return q, mask_loss

    def _update_mask_prototypes(self, task_id, scores):
      self.mmu.update_mask_prototype(task_id, scores)


class MinAtarDQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(MinAtarDQN, self).__init__()

        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        in_features = 8 * 8 * 16
        self.mmu = MaskedMemoryUnit(in_features=in_features, layer_size=256, mask_size=64, task_emb_size=32, num_tasks=5, out_features=128)

        self.v_head = nn.utils.spectral_norm(nn.Linear(in_features=128, out_features=1))
        self.adv_head = nn.utils.spectral_norm(nn.Linear(in_features=128, out_features=num_actions))

    def forward(self, x, task_id, update_mask_prototypes=False):
        x = x.permute(0,3,1,2)
        x = self.act(self.conv(x))
        x, mask_loss, scores = self.mmu(x.reshape(x.size(0), -1), task_id)
        x = self.act(x)

        v = self.v_head(x)
        adv = self.adv_head(x)
        q = v + (adv - adv.mean(dim=1, keepdim=True))

        if update_mask_prototypes == True:
          self._update_mask_prototypes(task_id, scores)

        return q, mask_loss

    def _update_mask_prototypes(self, task_id, scores):
      self.mmu.update_mask_prototype(task_id, scores)


"""
class MinatarVideoRecorder:
    def __init__(self, n_channels, resize_scale=20):
        self.n_channels = n_channels
        self.resize_scale = resize_scale
        palette = sns.color_palette("cubehelix", n_channels)
        self.colors = [(0, 0, 0)] + [(int(r*255), int(g*255), int(b*255)) for r, g, b in palette]
        self.frames = []

    def add_frame(self, state):
        weights = np.arange(1, self.n_channels + 1).reshape(1, 1, -1)
        weighted = state * weights
        max_vals = weighted.max(axis=2)
        argmax = np.argmax(weighted, axis=2)
        indices = np.where(max_vals > 0, argmax + 1, 0)  # 0 = background
        # Map to RGB
        frame = np.array(self.colors, dtype=np.uint8)[indices]  # shape (10,10,3)
        self.frames.append(frame)

    def save(self, filename="video.mp4", fps=10):
        if not self.frames:
            raise ValueError("No frames to record.")

        # Resize with nearest neighbour
        h, w = self.frames[0].shape[:2]
        new_h = h * self.resize_scale
        new_w = w * self.resize_scale
        # Ensure dimensions are even (required for yuv420p)
        if new_h % 2 != 0:
            new_h += 1
        if new_w % 2 != 0:
            new_w += 1

        resized = []
        for frame in self.frames:
            big = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized.append(big)

        imageio.mimsave(
            filename, resized, fps=fps,
            codec='libx264',
            quality=9,
            ffmpeg_params=[
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'high',
                '-tag:v', 'avc1'
            ]
        )

    def display(self, filename="video.mp4"):
        return Video(filename, width=400, embed=True)

    def clear(self):
        self.frames.clear()
"""