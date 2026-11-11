"""Dual-Stream hybrid batch sampler combining MCTS leaf negatives and uniform anchors."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import warnings

import numpy as np

from src.trees.base_tree import FastHierarchyTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualStreamBatch:
    r"""Mini-batch indices assembled by `DualStreamBatchSampler`.

    Attributes:
        leaf_node_id: MCTS-selected leaf node index in `FastHierarchyTree`.
        leaf_class_id: CIFAR-100 fine class ID associated with `leaf_node_id`.
        mcts_indices: 1D `int64` array of shape `[K_mcts]` sampled from `leaf_class_id`.
        uniform_indices: 1D `int64` array of shape `[K_uni]` sampled uniformly from all negatives.
        negative_indices: 1D `int64` array of shape `[B_neg]` (`concat(mcts_indices, uniform_indices)`).
        positive_indices: 1D `int64` array of shape `[B_pos]` (empty when `pos_batch_size == 0`).
        all_indices: 1D `int64` array of shape `[B_pos + B_neg]` (`concat(positive_indices, negative_indices)`).
        sampled_with_replacement: `True` if `leaf_class_id` had `< K_mcts` instances and required replacement.
    """

    leaf_node_id: int
    leaf_class_id: int
    mcts_indices: np.ndarray
    uniform_indices: np.ndarray
    negative_indices: np.ndarray
    positive_indices: np.ndarray
    all_indices: np.ndarray
    sampled_with_replacement: bool


class DualStreamBatchSampler:
    r"""Assemble negative mini-batches from MCTS-selected leaf samples and uniform anchor samples.

    For negative batch size $B_{\text{neg}}$ and uniform anchor ratio $\beta \in [0, 1]$:
    - $K_{\text{mcts}} = \lfloor (1 - \beta) B_{\text{neg}} \rfloor$
    - $K_{\text{uni}} = \lceil \beta B_{\text{neg}} \rceil$
    - $K_{\text{mcts}} + K_{\text{uni}} = B_{\text{neg}}$
    """

    def __init__(
        self,
        tree: FastHierarchyTree,
        neg_batch_size: int = 48,
        dual_stream_ratio: float = 0.5,
        positive_indices: np.ndarray | None = None,
        pos_batch_size: int = 0,
        seed: int = 42,
    ) -> None:
        if not isinstance(tree, FastHierarchyTree):
            raise ValueError(f"tree must be a FastHierarchyTree instance, got {type(tree).__name__}.")
        if isinstance(neg_batch_size, bool) or not isinstance(neg_batch_size, (int, np.integer)):
            raise ValueError(f"neg_batch_size must be a positive int, got {neg_batch_size!r}.")
        b_neg = int(neg_batch_size)
        if b_neg <= 0:
            raise ValueError(f"neg_batch_size must be >= 1, got {b_neg}.")

        if isinstance(dual_stream_ratio, bool) or not isinstance(
            dual_stream_ratio, (int, float, np.number)
        ):
            raise ValueError(f"dual_stream_ratio must be a float in [0.0, 1.0], got {dual_stream_ratio!r}.")
        beta = float(dual_stream_ratio)
        if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
            raise ValueError(f"dual_stream_ratio must be in [0.0, 1.0], got {beta}.")

        if isinstance(pos_batch_size, bool) or not isinstance(pos_batch_size, (int, np.integer)):
            raise ValueError(f"pos_batch_size must be a non-negative int, got {pos_batch_size!r}.")
        b_pos = int(pos_batch_size)
        if b_pos < 0:
            raise ValueError(f"pos_batch_size must be >= 0, got {b_pos}.")

        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer, got {seed!r}.")

        pos_pool: np.ndarray
        if b_pos > 0:
            if positive_indices is None:
                raise ValueError("positive_indices must be provided when pos_batch_size > 0.")
            pos_arr = np.asarray(positive_indices)
            if (
                pos_arr.ndim != 1
                or pos_arr.size == 0
                or not np.issubdtype(pos_arr.dtype, np.integer)
                or pos_arr.dtype == np.bool_
            ):
                raise ValueError("positive_indices must be a non-empty 1D integer array.")
            pos_pool = np.ascontiguousarray(pos_arr, dtype=np.int64)
        else:
            if positive_indices is not None:
                pos_arr = np.asarray(positive_indices)
                if pos_arr.ndim != 1 or not np.issubdtype(pos_arr.dtype, np.integer) or pos_arr.dtype == np.bool_:
                    raise ValueError("positive_indices must be a 1D integer array.")
                pos_pool = np.ascontiguousarray(pos_arr, dtype=np.int64)
            else:
                pos_pool = np.empty(0, dtype=np.int64)

        self.tree = tree
        self.neg_batch_size = b_neg
        self.dual_stream_ratio = beta
        self.pos_batch_size = b_pos
        self.positive_pool = pos_pool
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._warned_replacement = False

        # Compute exact floor((1 - beta) * B_neg) and ceil(beta * B_neg) with rounding guard
        mcts_exact = round((1.0 - beta) * b_neg, 9)
        uni_exact = round(beta * b_neg, 9)
        self.mcts_sub_batch_size = int(math.floor(mcts_exact))
        self.uniform_sub_batch_size = int(math.ceil(uni_exact))
        assert self.mcts_sub_batch_size + self.uniform_sub_batch_size == self.neg_batch_size

        sorted_cls_ids = sorted(self.tree.class_to_indices.keys())
        self.global_negative_pool = np.concatenate(
            [self.tree.class_to_indices[cid] for cid in sorted_cls_ids]
        ).astype(np.int64, copy=False)

    def sample_batch(self, leaf_node_id: int) -> DualStreamBatch:
        """Sample a hybrid negative (and optional positive) mini-batch for `leaf_node_id`."""
        if isinstance(leaf_node_id, bool) or not isinstance(leaf_node_id, (int, np.integer)):
            raise ValueError(f"leaf_node_id must be an int, got {leaf_node_id!r}.")
        lid = int(leaf_node_id)
        if lid < 0 or lid >= self.tree.num_nodes:
            raise ValueError(f"leaf_node_id {lid} out of range [0, {self.tree.num_nodes - 1}].")
        if not bool(self.tree.is_leaf[lid]):
            raise ValueError(f"sample_batch() requires a leaf node ID, got internal node {lid}.")

        leaf_class_id = int(self.tree.leaf_to_class_id[lid])
        leaf_pool = self.tree.class_to_indices[leaf_class_id]

        sampled_with_replacement = False
        k_mcts = self.mcts_sub_batch_size
        if k_mcts == 0:
            mcts_indices = np.empty(0, dtype=np.int64)
        elif leaf_pool.size >= k_mcts:
            mcts_indices = self._rng.choice(leaf_pool, size=k_mcts, replace=False).astype(
                np.int64, copy=False
            )
        else:
            sampled_with_replacement = True
            if not self._warned_replacement:
                msg = (
                    f"Leaf node {lid} (class {leaf_class_id}) has only {leaf_pool.size} instances "
                    f"(< requested K_mcts={k_mcts}); sampling with replacement."
                )
                logger.warning(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                self._warned_replacement = True
            mcts_indices = self._rng.choice(leaf_pool, size=k_mcts, replace=True).astype(
                np.int64, copy=False
            )

        k_uni = self.uniform_sub_batch_size
        if k_uni == 0:
            uniform_indices = np.empty(0, dtype=np.int64)
        else:
            replace_uni = self.global_negative_pool.size < k_uni
            uniform_indices = self._rng.choice(
                self.global_negative_pool, size=k_uni, replace=replace_uni
            ).astype(np.int64, copy=False)

        negative_indices = np.concatenate((mcts_indices, uniform_indices)).astype(
            np.int64, copy=False
        )

        if self.pos_batch_size > 0:
            replace_pos = self.positive_pool.size < self.pos_batch_size
            positive_indices = self._rng.choice(
                self.positive_pool, size=self.pos_batch_size, replace=replace_pos
            ).astype(np.int64, copy=False)
        else:
            positive_indices = np.empty(0, dtype=np.int64)

        all_indices = np.concatenate((positive_indices, negative_indices)).astype(
            np.int64, copy=False
        )

        return DualStreamBatch(
            leaf_node_id=lid,
            leaf_class_id=leaf_class_id,
            mcts_indices=mcts_indices,
            uniform_indices=uniform_indices,
            negative_indices=negative_indices,
            positive_indices=positive_indices,
            all_indices=all_indices,
            sampled_with_replacement=sampled_with_replacement,
        )


class UniformGroupedBatchSampler:
    r"""Uniform Grouped batch sampler (Phase C1 control).

    Samples a random focal class $c_{\text{focal}} \sim \text{Uniform}(\mathcal{C})$ at each step,
    drawing $K_{\text{focal}} = \lfloor (1 - \beta) B \rfloor$ from that class and
    $K_{\text{uni}} = \lceil \beta B \rceil$ uniformly from all classes.
    """

    def __init__(
        self,
        class_to_indices: dict[int, np.ndarray] | None = None,
        tree: FastHierarchyTree | None = None,
        batch_size: int = 64,
        dual_stream_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        if tree is not None:
            c2i = tree.class_to_indices
        elif class_to_indices is not None:
            c2i = class_to_indices
        else:
            raise ValueError("Either tree or class_to_indices must be provided.")

        if isinstance(dual_stream_ratio, bool) or not isinstance(
            dual_stream_ratio, (int, float, np.number)
        ):
            raise ValueError(f"dual_stream_ratio must be a float in [0.0, 1.0], got {dual_stream_ratio!r}.")
        beta = float(dual_stream_ratio)
        if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
            raise ValueError(f"dual_stream_ratio must be in [0.0, 1.0], got {beta}.")

        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}.")
        b = int(batch_size)
        if b <= 0:
            raise ValueError(f"batch_size must be >= 1, got {b}.")

        self.class_to_indices = {int(k): np.asarray(v, dtype=np.int64) for k, v in c2i.items()}
        self.classes = sorted(self.class_to_indices.keys())
        if len(self.classes) == 0:
            raise ValueError("class_to_indices must be non-empty.")

        self.batch_size = b
        self.dual_stream_ratio = beta
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        mcts_exact = round((1.0 - beta) * b, 9)
        uni_exact = round(beta * b, 9)
        self.focal_sub_batch_size = int(math.floor(mcts_exact))
        self.uniform_sub_batch_size = int(math.ceil(uni_exact))
        assert self.focal_sub_batch_size + self.uniform_sub_batch_size == self.batch_size

        self.global_pool = np.concatenate(
            [self.class_to_indices[cid] for cid in self.classes]
        ).astype(np.int64, copy=False)

    def sample_batch(self) -> DualStreamBatch:
        """Sample a batch of focal samples from a uniformly random class and uniform background samples."""
        focal_class = int(self._rng.choice(self.classes))
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

