"""Canonical CIFAR-100 20-superclass <-> 100-fine-class taxonomy mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
import torch

CIFAR100_FINE_CLASSES: tuple[str, ...] = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    " cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)
# Ensure canonical stripped names (fixing any leading whitespace)
CIFAR100_FINE_CLASSES = tuple(name.strip() for name in CIFAR100_FINE_CLASSES)

CIFAR100_COARSE_CLASSES: tuple[str, ...] = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)

SUPERCLASS_TO_FINE_NAMES: dict[str, tuple[str, ...]] = {
    "aquatic_mammals": ("beaver", "dolphin", "otter", "seal", "whale"),
    "fish": ("aquarium_fish", "flatfish", "ray", "shark", "trout"),
    "flowers": ("orchid", "poppy", "rose", "sunflower", "tulip"),
    "food_containers": ("bottle", "bowl", "can", "cup", "plate"),
    "fruit_and_vegetables": ("apple", "mushroom", "orange", "pear", "sweet_pepper"),
    "household_electrical_devices": ("clock", "keyboard", "lamp", "telephone", "television"),
    "household_furniture": ("bed", "chair", "couch", "table", "wardrobe"),
    "insects": ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
    "large_carnivores": ("bear", "leopard", "lion", "tiger", "wolf"),
    "large_man-made_outdoor_things": ("bridge", "castle", "house", "road", "skyscraper"),
    "large_natural_outdoor_scenes": ("cloud", "forest", "mountain", "plain", "sea"),
    "large_omnivores_and_herbivores": ("camel", "cattle", "chimpanzee", "elephant", "kangaroo"),
    "medium_mammals": ("fox", "porcupine", "possum", "raccoon", "skunk"),
    "non-insect_invertebrates": ("crab", "lobster", "snail", "spider", "worm"),
    "people": ("baby", "boy", "girl", "man", "woman"),
    "reptiles": ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
    "small_mammals": ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
    "trees": ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"),
    "vehicles_1": ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
    "vehicles_2": ("lawn_mower", "rocket", "streetcar", "tank", "tractor"),
}


@dataclass(frozen=True)
class CIFAR100Taxonomy:
    """Immutable CIFAR-100 20-superclass <-> 100-fine-class taxonomy structure."""

    fine_names: tuple[str, ...]
    coarse_names: tuple[str, ...]
    fine_to_coarse_idx: np.ndarray
    coarse_to_fine_ids: dict[int, tuple[int, ...]]
    fine_name_to_id: dict[str, int]
    coarse_name_to_id: dict[str, int]


def get_cifar100_taxonomy() -> CIFAR100Taxonomy:
    """Build and return the verified CIFAR-100 taxonomy mapping."""
    fine_name_to_id = {name: idx for idx, name in enumerate(CIFAR100_FINE_CLASSES)}
    coarse_name_to_id = {name: idx for idx, name in enumerate(CIFAR100_COARSE_CLASSES)}

    fine_to_coarse = np.full(len(CIFAR100_FINE_CLASSES), -1, dtype=np.int64)
    coarse_to_fine_ids: dict[int, tuple[int, ...]] = {}

    for coarse_idx, coarse_name in enumerate(CIFAR100_COARSE_CLASSES):
        fine_group = SUPERCLASS_TO_FINE_NAMES[coarse_name]
        fine_ids = tuple(sorted(fine_name_to_id[fname] for fname in fine_group))
        coarse_to_fine_ids[coarse_idx] = fine_ids
        for fid in fine_ids:
            fine_to_coarse[fid] = coarse_idx

    fine_to_coarse.flags.writeable = False
    return CIFAR100Taxonomy(
        fine_names=CIFAR100_FINE_CLASSES,
        coarse_names=CIFAR100_COARSE_CLASSES,
        fine_to_coarse_idx=fine_to_coarse,
        coarse_to_fine_ids=coarse_to_fine_ids,
        fine_name_to_id=fine_name_to_id,
        coarse_name_to_id=coarse_name_to_id,
    )


_TAXONOMY = get_cifar100_taxonomy()


@overload
def fine_to_coarse_tensor(fine_labels: torch.Tensor) -> torch.Tensor: ...


@overload
def fine_to_coarse_tensor(fine_labels: np.ndarray) -> np.ndarray: ...


def fine_to_coarse_tensor(fine_labels: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Map CIFAR-100 fine class indices (`0..99`) to coarse superclass indices (`0..19`).

    Args:
        fine_labels: Tensor or NumPy array of integer fine labels in `[0, 99]`.

    Returns:
        Coarse superclass indices in `[0, 19]` matching the input type, shape, and device.

    Raises:
        ValueError: If any label is outside `[0, 99]` or input is not an integer tensor/array.
    """
    if isinstance(fine_labels, torch.Tensor):
        if fine_labels.dtype == torch.bool or torch.is_floating_point(fine_labels) or torch.is_complex(fine_labels):
            raise ValueError(f"fine_labels must be an integer tensor, got dtype={fine_labels.dtype}.")
        if fine_labels.numel() > 0:
            min_val = int(fine_labels.min().item())
            max_val = int(fine_labels.max().item())
            if min_val < 0 or max_val >= 100:
                raise ValueError(
                    f"CIFAR-100 fine labels must be in [0, 99], got range [{min_val}, {max_val}]."
                )
        lut = torch.tensor(
            _TAXONOMY.fine_to_coarse_idx.tolist(), dtype=torch.long, device=fine_labels.device
        )
        return lut[fine_labels.long()]

    if isinstance(fine_labels, np.ndarray):
        if not np.issubdtype(fine_labels.dtype, np.integer) or fine_labels.dtype == np.bool_:
            raise ValueError(f"fine_labels must be an integer ndarray, got dtype={fine_labels.dtype}.")
        if fine_labels.size > 0:
            min_val = int(fine_labels.min())
            max_val = int(fine_labels.max())
            if min_val < 0 or max_val >= 100:
                raise ValueError(
                    f"CIFAR-100 fine labels must be in [0, 99], got range [{min_val}, {max_val}]."
                )
        return _TAXONOMY.fine_to_coarse_idx[fine_labels.astype(np.int64)].copy()

    raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(fine_labels).__name__}.")
