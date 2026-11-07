import numpy as np
from collections.abc import Callable
import operator
import random

class ReplayBuffer:
    def __init__(self, obs_shape, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.uint8)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.uint8)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.terminated_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        terminated: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.terminated_buf[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            terminated=self.terminated_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size

    def clear(self):
        self.obs_buf.fill(0)
        self.next_obs_buf.fill(0)
        self.acts_buf.fill(0)
        self.rews_buf.fill(0)
        self.terminated_buf.fill(0)
        self.ptr = 0
        self.size = 0


class SegmentTree:
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert capacity > 0 and capacity & (capacity - 1) == 0, (
            "capacity must be positive and a power of 2."
        )
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        elif mid + 1 <= start:
            return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self.operation(
                self._operate_helper(start, mid, 2 * node, node_start, mid),
                self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
            )

    def operate(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        assert 0 <= idx < self.capacity
        return self.tree[self.capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        return super().operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        assert 0 <= upperbound <= self.sum() + 1e-5, f"upperbound: {upperbound}"
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        return super().operate(start, end)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, size: int, batch_size: int = 32, alpha: float = 0.6):
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_shape, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be power of 2
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, terminated: bool):
        super().store(obs, act, rew, next_obs, terminated)

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> dict[str, np.ndarray]:
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        terminated = self.terminated_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            terminated=terminated,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: list[int], priorities: np.ndarray):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> list[int]:
        indices = []
        p_total = self.sum_tree.sum(0, len(self))
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idx = min(idx, len(self) - 1)  # weird bug fix
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def clear(self):
        super().clear()

        self.max_priority = 1.0
        self.tree_ptr = 0

        tree_capacity = self.sum_tree.capacity
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)