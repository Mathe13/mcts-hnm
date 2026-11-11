"""Simple baseline selectors for active curriculum ablations (Phase C3).

Includes:
- GreedyLossEMASelector: Greedy class selection based on EMA of class loss (no UCB, c=0).
- FlatUCBSelector: Flat multi-armed bandit UCB1 over classes (no tree hierarchy).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.mcts.sampler import DualStreamBatch


class GreedyLossEMASelector:
    r"""Greedy selector by exponential moving average (EMA) of class loss (Phase C3).

    Maintains an exponential moving average $\bar{\ell}_c$ for each class $c \in \{0, \dots, K-1\}$:
    $$\bar{\ell}_c \leftarrow (1 - \alpha) \bar{\ell}_c + \alpha \ell_c$$
    and greedily selects the class with highest loss:
    $$c^* = \arg\max_c \bar{\ell}_c$$
    without any exploration bonus ($c = 0$) or tree hierarchy.
    """

    def __init__(
        self,
        class_to_indices: dict[int, np.ndarray],
        num_classes: int = 100,
        batch_size: int = 64,
        dual_stream_ratio: float = 0.5,
        alpha: float = 0.3,
        seed: int = 42,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if not (0.0 <= dual_stream_ratio <= 1.0):
            raise ValueError(f"dual_stream_ratio must be in [0.0, 1.0], got {dual_stream_ratio}.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}.")

        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.dual_stream_ratio = float(dual_stream_ratio)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.class_to_indices = {int(k): np.asarray(v, dtype=np.int64) for k, v in class_to_indices.items()}
        self.class_ema_loss = np.zeros(self.num_classes, dtype=np.float64)

        mcts_exact = round((1.0 - self.dual_stream_ratio) * self.batch_size, 9)
        uni_exact = round(self.dual_stream_ratio * self.batch_size, 9)
        self.focal_sub_batch_size = int(math.floor(mcts_exact))
        self.uniform_sub_batch_size = int(math.ceil(uni_exact))

        sorted_cls = sorted(self.class_to_indices.keys())
        self.global_pool = np.concatenate([self.class_to_indices[cid] for cid in sorted_cls]).astype(
            np.int64, copy=False
        )

    def select_class(self) -> int:
        """Select class with highest EMA loss (ties broken randomly)."""
        max_loss = np.max(self.class_ema_loss)
        candidates = np.where(self.class_ema_loss == max_loss)[0]
        if len(candidates) == 1:
            return int(candidates[0])
        return int(self._rng.choice(candidates))

    def sample_batch(self) -> DualStreamBatch:
        """Sample a batch focused on the greedily selected class."""
        focal_class = self.select_class()
        focal_pool = self.class_to_indices[focal_class]

        k_focal = self.focal_sub_batch_size
        sampled_with_replacement = False
        if k_focal == 0:
            focal_indices = np.empty(0, dtype=np.int64)
        elif focal_pool.size >= k_focal:
            focal_indices = self._rng.choice(focal_pool, size=k_focal, replace=False).astype(
                np.int64, copy=False
            )
        else:
            sampled_with_replacement = True
            focal_indices = self._rng.choice(focal_pool, size=k_focal, replace=True).astype(
                np.int64, copy=False
            )

        k_uni = self.uniform_sub_batch_size
        if k_uni == 0:
            uni_indices = np.empty(0, dtype=np.int64)
        else:
            replace_uni = self.global_pool.size < k_uni
            uni_indices = self._rng.choice(self.global_pool, size=k_uni, replace=replace_uni).astype(
                np.int64, copy=False
            )

        all_indices = np.concatenate((focal_indices, uni_indices)).astype(np.int64, copy=False)

        return DualStreamBatch(
            leaf_node_id=-1,
            leaf_class_id=focal_class,
            mcts_indices=focal_indices,
            uniform_indices=uni_indices,
            negative_indices=all_indices,
            positive_indices=np.empty(0, dtype=np.int64),
            all_indices=all_indices,
            sampled_with_replacement=sampled_with_replacement,
        )

    def update(self, class_id: int, observed_loss: float) -> None:
        """Update EMA loss for `class_id`."""
        cid = int(class_id)
        if 0 <= cid < self.num_classes:
            loss_val = float(observed_loss)
            self.class_ema_loss[cid] = (1.0 - self.alpha) * self.class_ema_loss[cid] + self.alpha * loss_val


class FlatUCBSelector:
    r"""Flat UCB1 multi-armed bandit selector over classes without hierarchy (Phase C3).

    Treats each class $c \in \{0, \dots, K-1\}$ as an independent bandit arm with visits $n_c$
    and action-value $Q_c$.
    - Unvisited classes have infinite priority ($\infty$).
    - Visited classes are selected via UCB1:
      $$c^* = \arg\max_c \left[ Q_c + c_{\text{ucb}} \sqrt{\frac{\ln N}{n_c}} \right]$$
      where $N = \sum_{j=1}^K n_j$.
    """

    def __init__(
        self,
        class_to_indices: dict[int, np.ndarray],
        num_classes: int = 100,
        batch_size: int = 64,
        dual_stream_ratio: float = 0.5,
        exploration_c: float = 2.0,
        seed: int = 42,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if not (0.0 <= dual_stream_ratio <= 1.0):
            raise ValueError(f"dual_stream_ratio must be in [0.0, 1.0], got {dual_stream_ratio}.")
        if exploration_c < 0.0:
            raise ValueError(f"exploration_c must be >= 0.0, got {exploration_c}.")

        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.dual_stream_ratio = float(dual_stream_ratio)
        self.exploration_c = float(exploration_c)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.class_to_indices = {int(k): np.asarray(v, dtype=np.int64) for k, v in class_to_indices.items()}
        self.visits = np.zeros(self.num_classes, dtype=np.int64)
        self.q_values = np.zeros(self.num_classes, dtype=np.float64)

        mcts_exact = round((1.0 - self.dual_stream_ratio) * self.batch_size, 9)
        uni_exact = round(self.dual_stream_ratio * self.batch_size, 9)
        self.focal_sub_batch_size = int(math.floor(mcts_exact))
        self.uniform_sub_batch_size = int(math.ceil(uni_exact))

        sorted_cls = sorted(self.class_to_indices.keys())
        self.global_pool = np.concatenate([self.class_to_indices[cid] for cid in sorted_cls]).astype(
            np.int64, copy=False
        )

    def select_class(self) -> int:
        """Select class using UCB1 with zero-visit priority."""
        unvisited = np.where(self.visits == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0])

        total_visits = int(np.sum(self.visits))
        log_n = math.log(max(total_visits, 1))
        exploration = self.exploration_c * np.sqrt(log_n / self.visits.astype(np.float64))
        ucb_scores = self.q_values + exploration

        max_score = np.max(ucb_scores)
        candidates = np.where(ucb_scores == max_score)[0]
        if len(candidates) == 1:
            return int(candidates[0])
        return int(self._rng.choice(candidates))

    def sample_batch(self) -> DualStreamBatch:
        """Sample a batch focused on the UCB-selected class."""
        focal_class = self.select_class()
        focal_pool = self.class_to_indices[focal_class]

        k_focal = self.focal_sub_batch_size
        sampled_with_replacement = False
        if k_focal == 0:
            focal_indices = np.empty(0, dtype=np.int64)
        elif focal_pool.size >= k_focal:
            focal_indices = self._rng.choice(focal_pool, size=k_focal, replace=False).astype(
                np.int64, copy=False
            )
        else:
            sampled_with_replacement = True
            focal_indices = self._rng.choice(focal_pool, size=k_focal, replace=True).astype(
                np.int64, copy=False
            )

        k_uni = self.uniform_sub_batch_size
        if k_uni == 0:
            uni_indices = np.empty(0, dtype=np.int64)
        else:
            replace_uni = self.global_pool.size < k_uni
            uni_indices = self._rng.choice(self.global_pool, size=k_uni, replace=replace_uni).astype(
                np.int64, copy=False
            )

        all_indices = np.concatenate((focal_indices, uni_indices)).astype(np.int64, copy=False)

        return DualStreamBatch(
            leaf_node_id=-1,
            leaf_class_id=focal_class,
            mcts_indices=focal_indices,
            uniform_indices=uni_indices,
            negative_indices=all_indices,
            positive_indices=np.empty(0, dtype=np.int64),
            all_indices=all_indices,
            sampled_with_replacement=sampled_with_replacement,
        )

    def update(self, class_id: int, reward: float) -> None:
        """Update sample-average action-value Q and visit count for `class_id`."""
        cid = int(class_id)
        if 0 <= cid < self.num_classes:
            self.visits[cid] += 1
            n = self.visits[cid]
            r = float(reward)
            # Incremental sample average: Q_{n} = Q_{n-1} + (R - Q_{n-1}) / n
            self.q_values[cid] += (r - self.q_values[cid]) / float(n)
