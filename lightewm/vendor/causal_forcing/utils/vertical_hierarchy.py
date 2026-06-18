from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


CONDITION_TOKEN_ID = -1


def _dynamic_budget_from_cover(covered_frames: float, max_step_budget: int = 50) -> int:
    """
    Map "covered latent frames" -> denoising budget using log-space piecewise interpolation.
    Anchor points (cover -> budget): 21->7, 10.5->11, 5.25->17, 2.675->26, 1.8->40, 1->50.
    For covers outside [1, 21], we extrapolate using the nearest edge segment.
    """
    if covered_frames <= 0:
        raise ValueError(f"covered_frames must be positive, got {covered_frames}.")
    if max_step_budget <= 0:
        raise ValueError(f"max_step_budget must be positive, got {max_step_budget}.")

    anchor_cover = [1.0, 1.8, 2.675, 5.25, 10.5, 21.0]
    anchor_budget = [50.0, 40.0, 26.0, 17.0, 11.0, 7.0]

    # Scale anchors when max budget != 50 while preserving relative profile.
    if max_step_budget != 50:
        scale = max_step_budget / 50.0
        anchor_budget = [max(1.0, min(float(max_step_budget), b * scale)) for b in anchor_budget]

    x = float(covered_frames)
    if x <= anchor_cover[0]:
        left_index = 0
    elif x >= anchor_cover[-1]:
        left_index = len(anchor_cover) - 2
    else:
        left_index = 0
        while not (anchor_cover[left_index] <= x <= anchor_cover[left_index + 1]):
            left_index += 1

    x1 = anchor_cover[left_index]
    x2 = anchor_cover[left_index + 1]
    y1 = anchor_budget[left_index]
    y2 = anchor_budget[left_index + 1]

    t = (math.log(x) - math.log(x1)) / (math.log(x2) - math.log(x1))
    interpolated = math.exp(math.log(y1) + t * (math.log(y2) - math.log(y1)))
    budget = int(round(interpolated))
    return max(1, min(max_step_budget, budget))


def build_vertical_hierarchy(
    num_leaf_frames: int = 21,
    num_levels: Optional[int] = 6,
    level_sizes: Optional[List[int]] = None,
    start_from_st_end: bool = False,
    st_end_plus: bool = False,
    allow_condition_for_all_frames: bool = False,
) -> Dict[str, List[int] | List[Tuple[int, int]] | List[List[Tuple[int, int]]]]:
    if num_leaf_frames <= 0:
        raise ValueError(f"num_leaf_frames must be positive, got {num_leaf_frames}.")
    if num_levels is not None and num_levels <= 0:
        raise ValueError(f"num_levels must be positive, got {num_levels}.")
    if start_from_st_end and st_end_plus:
        raise ValueError("start_from_st_end and st_end_plus are mutually exclusive; enable at most one.")

    if level_sizes is None:
        if start_from_st_end:
            if num_levels is None:
                level_sizes = [1] if num_leaf_frames == 1 else [2]
                current_size = level_sizes[-1]
                while current_size < num_leaf_frames:
                    current_size = min(num_leaf_frames, current_size * 2)
                    level_sizes.append(current_size)
                num_levels = len(level_sizes)
            else:
                level_sizes = [1] if num_leaf_frames == 1 else [2]
                current_size = level_sizes[-1]
                for _ in range(1, num_levels):
                    current_size = min(num_leaf_frames, current_size * 2)
                    level_sizes.append(current_size)
        else:
            if num_levels is None:
                num_levels = 1
                current_size = 1
                while current_size < num_leaf_frames:
                    current_size = min(num_leaf_frames, current_size * 2)
                    num_levels += 1
            level_sizes = [1]
            current_size = 1
            for _ in range(1, num_levels):
                current_size = min(num_leaf_frames, current_size * 2)
                level_sizes.append(current_size)
    elif len(level_sizes) != num_levels:
        raise ValueError(f"Expected {num_levels} level_sizes, got {len(level_sizes)}.")

    if start_from_st_end:
        positions_per_level: List[List[int]] = []
        for level_size in level_sizes:
            if level_size <= 0:
                raise ValueError(f"level_size must be positive, got {level_size}.")
            positions = torch.linspace(0, num_leaf_frames - 1, steps=level_size, dtype=torch.float64)
            positions = torch.round(positions).to(dtype=torch.long).tolist()
            if len(set(positions)) != level_size:
                raise ValueError(
                    f"start_from_st_end produced duplicate representative indices for level_size={level_size}: {positions}"
                )
            positions_per_level.append(positions)

        levels = [[(position, position) for position in level_positions] for level_positions in positions_per_level]
        level_token_ids: List[List[int]] = []
        token_to_parent: List[int] = []
        next_token_id = 0
        for level_positions in positions_per_level:
            current_token_ids = list(range(next_token_id, next_token_id + len(level_positions)))
            level_token_ids.append(current_token_ids)
            next_token_id += len(level_positions)

        token_to_parent = [-1] * next_token_id
        for level_index in range(1, len(positions_per_level)):
            parent_positions = positions_per_level[level_index - 1]
            parent_token_ids = level_token_ids[level_index - 1]
            current_positions = positions_per_level[level_index]
            current_token_ids = level_token_ids[level_index]
            for token_id, position in zip(current_token_ids, current_positions):
                nearest_parent_index = min(
                    range(len(parent_positions)),
                    key=lambda parent_index: abs(parent_positions[parent_index] - position),
                )
                token_to_parent[token_id] = parent_token_ids[nearest_parent_index]
    else:
        levels = [[(0, num_leaf_frames - 1)]]
        level_token_ids = [[0]]
        token_to_parent = [-1]
        next_token_id = 1

        for _ in range(1, num_levels):
            prev_level = levels[-1]
            prev_token_ids = level_token_ids[-1]
            current_level: List[Tuple[int, int]] = []
            current_token_ids: List[int] = []
            for parent_token_id, (start, end) in zip(prev_token_ids, prev_level):
                if start < end:
                    mid = (start + end) // 2
                    children = [(start, mid), (mid + 1, end)]
                else:
                    children = [(start, end)]

                for child in children:
                    current_level.append(child)
                    current_token_ids.append(next_token_id)
                    token_to_parent.append(parent_token_id)
                    next_token_id += 1

            levels.append(current_level)
            level_token_ids.append(current_token_ids)

        if st_end_plus:
            base_token_to_level_pos = {}
            for level_index, token_ids in enumerate(level_token_ids):
                for level_pos, token_id in enumerate(token_ids):
                    base_token_to_level_pos[token_id] = (level_index, level_pos)

            augmented_levels: List[List[Tuple[int, int]]] = []
            augmented_level_token_ids: List[List[int]] = []
            augmented_token_to_parent: List[int] = []
            next_token_id = 0
            previous_level_mid_pos_to_token_id: Dict[int, int] = {}
            previous_level_st_token_id = -1
            previous_level_ed_token_id = -1
            last_leaf_index = num_leaf_frames - 1

            for level_index, base_level in enumerate(levels):
                base_representative_indices = [(interval[0] + interval[1]) // 2 for interval in base_level]
                add_st_token = 0 not in base_representative_indices
                add_ed_token = last_leaf_index not in base_representative_indices

                augmented_level: List[Tuple[int, int]] = []
                augmented_token_ids: List[int] = []
                current_level_mid_pos_to_token_id: Dict[int, int] = {}

                if add_st_token:
                    st_token_id = next_token_id
                    next_token_id += 1
                    augmented_level.append((0, 0))
                    augmented_token_ids.append(st_token_id)
                    augmented_token_to_parent.append(previous_level_st_token_id if level_index > 0 else -1)

                for base_pos, interval in enumerate(base_level):
                    token_id = next_token_id
                    next_token_id += 1
                    augmented_level.append(interval)
                    augmented_token_ids.append(token_id)
                    current_level_mid_pos_to_token_id[base_pos] = token_id

                    if level_index == 0:
                        parent_token_id = -1
                    else:
                        base_token_id = level_token_ids[level_index][base_pos]
                        base_parent_token_id = token_to_parent[base_token_id]
                        if base_parent_token_id < 0:
                            parent_token_id = -1
                        else:
                            parent_level_index, parent_level_pos = base_token_to_level_pos[base_parent_token_id]
                            if parent_level_index != level_index - 1:
                                raise RuntimeError("Invalid base vertical hierarchy parent relationship.")
                            parent_token_id = previous_level_mid_pos_to_token_id[parent_level_pos]
                    augmented_token_to_parent.append(parent_token_id)

                if add_ed_token:
                    ed_token_id = next_token_id
                    next_token_id += 1
                    augmented_level.append((last_leaf_index, last_leaf_index))
                    augmented_token_ids.append(ed_token_id)
                    augmented_token_to_parent.append(previous_level_ed_token_id if level_index > 0 else -1)

                if add_st_token:
                    current_level_st_token_id = augmented_token_ids[0]
                else:
                    st_mid_positions = [
                        base_pos for base_pos, rep_index in enumerate(base_representative_indices) if rep_index == 0
                    ]
                    if not st_mid_positions:
                        raise RuntimeError("st_end_plus expected a start representative token but found none.")
                    current_level_st_token_id = current_level_mid_pos_to_token_id[st_mid_positions[0]]

                if add_ed_token:
                    current_level_ed_token_id = augmented_token_ids[-1]
                else:
                    ed_mid_positions = [
                        base_pos
                        for base_pos, rep_index in enumerate(base_representative_indices)
                        if rep_index == last_leaf_index
                    ]
                    if not ed_mid_positions:
                        raise RuntimeError("st_end_plus expected an end representative token but found none.")
                    current_level_ed_token_id = current_level_mid_pos_to_token_id[ed_mid_positions[0]]

                previous_level_st_token_id = current_level_st_token_id
                previous_level_ed_token_id = current_level_ed_token_id
                previous_level_mid_pos_to_token_id = current_level_mid_pos_to_token_id
                augmented_levels.append(augmented_level)
                augmented_level_token_ids.append(augmented_token_ids)

            levels = augmented_levels
            level_token_ids = augmented_level_token_ids
            token_to_parent = augmented_token_to_parent

    level_sizes = [len(level) for level in levels]
    level_offsets: List[int] = []
    running_offset = 0
    for size in level_sizes:
        level_offsets.append(running_offset)
        running_offset += size

    representative_indices: List[int] = []
    token_to_level = [0] * running_offset
    token_to_level_pos = [0] * running_offset
    token_to_interval: List[Tuple[int, int]] = [(0, 0)] * running_offset

    for level_index, level in enumerate(levels):
        for position_in_level, interval in enumerate(level):
            token_id = level_offsets[level_index] + position_in_level
            representative_indices.append((interval[0] + interval[1]) // 2)
            token_to_level[token_id] = level_index
            token_to_level_pos[token_id] = position_in_level
            token_to_interval[token_id] = interval

    if len(representative_indices) != running_offset:
        raise RuntimeError("Vertical hierarchy token count mismatch.")

    leaf_start_index = level_offsets[-1]
    leaf_token_ids = list(range(leaf_start_index, leaf_start_index + level_sizes[-1]))

    return {
        "num_leaf_frames": num_leaf_frames,
        "num_levels": num_levels,
        "levels": levels,
        "level_sizes": level_sizes,
        "level_offsets": level_offsets,
        "representative_indices": representative_indices,
        "token_to_parent": token_to_parent,
        "token_to_level": token_to_level,
        "token_to_level_pos": token_to_level_pos,
        "token_to_interval": token_to_interval,
        "num_tokens": running_offset,
        "leaf_start_index": leaf_start_index,
        "leaf_token_ids": leaf_token_ids,
        "allow_condition_for_all_frames": allow_condition_for_all_frames,
    }


def gather_vertical_latents(clean_latent: torch.Tensor, hierarchy: Dict[str, List[int]]) -> torch.Tensor:
    if clean_latent.ndim != 5:
        raise ValueError(f"Expected clean_latent to have shape [B, F, C, H, W], got {tuple(clean_latent.shape)}.")
    rep_indices = torch.tensor(
        hierarchy["representative_indices"],
        device=clean_latent.device,
        dtype=torch.long,
    )
    return clean_latent.index_select(dim=1, index=rep_indices)


def get_vertical_leaf_latents(vertical_latents: torch.Tensor, hierarchy: Dict[str, List[int]]) -> torch.Tensor:
    leaf_start = hierarchy["leaf_start_index"]
    return vertical_latents[:, leaf_start:]


def get_vertical_token_step_budgets(
    hierarchy: Dict[str, List[int]],
    level_step_budgets: List[int],
) -> List[int]:
    level_sizes = hierarchy["level_sizes"]
    if len(level_step_budgets) != len(level_sizes):
        raise ValueError(
            f"Expected {len(level_sizes)} vertical_step_budgets, got {len(level_step_budgets)}."
        )
    token_budgets: List[int] = []
    for size, budget in zip(level_sizes, level_step_budgets):
        token_budgets.extend([budget] * size)
    return token_budgets


def get_dynamic_vertical_token_step_budgets(
    hierarchy: Dict[str, List[int]],
    max_step_budget: int = 50,
) -> List[int]:
    token_budgets: List[int] = []
    for start, end in hierarchy["token_to_interval"]:
        covered_frames = end - start + 1
        token_budgets.append(_dynamic_budget_from_cover(float(covered_frames), max_step_budget))
    return token_budgets


def get_dynamic_vertical_level_avg_step_budgets(
    hierarchy: Dict[str, List[int]],
    max_step_budget: int = 50,
) -> List[int]:
    token_budgets: List[int] = []
    num_leaf_frames = hierarchy["num_leaf_frames"]
    for level_size in hierarchy["level_sizes"]:
        avg_cover = num_leaf_frames / level_size
        budget = _dynamic_budget_from_cover(float(avg_cover), max_step_budget)
        token_budgets.extend([budget] * level_size)
    return token_budgets


def vertical_token_id_to_sequence_position(token_id: int) -> int:
    return 0 if token_id == CONDITION_TOKEN_ID else token_id + 1


def get_vertical_allowed_token_ids(
    current_token_id: int,
    hierarchy: Dict[str, List[int]],
    *,
    include_self: bool = True,
    condition_token_id: int = CONDITION_TOKEN_ID,
) -> List[int]:
    if current_token_id == condition_token_id:
        return [condition_token_id] if include_self else []

    token_to_parent = hierarchy["token_to_parent"]
    token_to_level = hierarchy["token_to_level"]
    token_to_level_pos = hierarchy["token_to_level_pos"]
    level_offsets = hierarchy["level_offsets"]
    level_sizes = hierarchy["level_sizes"]
    allow_condition_for_all_frames = bool(hierarchy.get("allow_condition_for_all_frames", False))

    allowed_token_ids = set()
    if include_self:
        allowed_token_ids.add(current_token_id)

    current_level = token_to_level[current_token_id]
    current_level_pos = token_to_level_pos[current_token_id]
    if current_level_pos > 0:
        allowed_token_ids.add(level_offsets[current_level] + current_level_pos - 1)
    else:
        allowed_token_ids.add(condition_token_id)
    if allow_condition_for_all_frames:
        allowed_token_ids.add(condition_token_id)

    parent_token_id = token_to_parent[current_token_id]
    if parent_token_id >= 0:
        allowed_token_ids.add(parent_token_id)
        parent_level = token_to_level[parent_token_id]
        parent_level_pos = token_to_level_pos[parent_token_id]
        if parent_level_pos > 0:
            allowed_token_ids.add(parent_token_id - 1)
        if parent_level_pos + 1 < level_sizes[parent_level]:
            allowed_token_ids.add(parent_token_id + 1)

    return sorted(allowed_token_ids, key=vertical_token_id_to_sequence_position)
