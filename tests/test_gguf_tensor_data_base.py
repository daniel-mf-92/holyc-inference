#!/usr/bin/env python3
"""Reference checks for GGUF tensor-data base alignment and offset semantics."""

from __future__ import annotations

import random

GGUF_TDBASE_OK = 0
GGUF_TDBASE_ERR_NULL_PTR = 1
GGUF_TDBASE_ERR_BAD_ALIGNMENT = 2
GGUF_TDBASE_ERR_OVERFLOW = 3
GGUF_TDBASE_ERR_MISALIGNED_BASE = 4
GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET = 5
GGUF_TDBASE_ERR_BAD_TYPE = 6
GGUF_TDBASE_ERR_OUT_OF_BOUNDS = 7
GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE = 8
GGUF_TDBASE_ERR_OVERLAP = 9

GGUF_DEFAULT_ALIGNMENT = 32
U64_MAX = (1 << 64) - 1

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


def tensor_type_block_size(ggml_type: int):
    if ggml_type == GGML_TYPE_F32:
        return GGUF_TDBASE_OK, 1
    if ggml_type == GGML_TYPE_F16:
        return GGUF_TDBASE_OK, 1
    if ggml_type == GGML_TYPE_Q4_0:
        return GGUF_TDBASE_OK, 32
    if ggml_type == GGML_TYPE_Q8_0:
        return GGUF_TDBASE_OK, 32
    return GGUF_TDBASE_ERR_BAD_TYPE, 0


def tensor_type_block_bytes(ggml_type: int):
    if ggml_type == GGML_TYPE_F32:
        return GGUF_TDBASE_OK, 4
    if ggml_type == GGML_TYPE_F16:
        return GGUF_TDBASE_OK, 2
    if ggml_type == GGML_TYPE_Q4_0:
        return GGUF_TDBASE_OK, 18
    if ggml_type == GGML_TYPE_Q8_0:
        return GGUF_TDBASE_OK, 34
    return GGUF_TDBASE_ERR_BAD_TYPE, 0


def is_pow2_u32(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def tensor_bytes_for_type(ggml_type: int, element_count: int):
    err, block_size = tensor_type_block_size(ggml_type)
    if err != GGUF_TDBASE_OK:
        return err, 0

    err, block_bytes = tensor_type_block_bytes(ggml_type)
    if err != GGUF_TDBASE_OK:
        return err, 0

    if element_count % block_size != 0:
        return GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0

    block_count = element_count // block_size
    if block_count > U64_MAX // block_bytes:
        return GGUF_TDBASE_ERR_OVERFLOW, 0

    return GGUF_TDBASE_OK, block_count * block_bytes


def validate_alignment(alignment: int) -> int:
    if not is_pow2_u32(alignment):
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT
    return GGUF_TDBASE_OK


def align_up_u64(value: int, alignment: int):
    err = validate_alignment(alignment)
    if err != GGUF_TDBASE_OK:
        return err, 0

    mask = alignment - 1
    if value > U64_MAX - mask:
        return GGUF_TDBASE_ERR_OVERFLOW, 0

    out = (value + mask) & ~mask
    return GGUF_TDBASE_OK, out


def tensor_data_base_align(cursor_after_tensor_info: int, alignment: int):
    return align_up_u64(cursor_after_tensor_info, alignment)


def tensor_data_base_align_default(cursor_after_tensor_info: int):
    return tensor_data_base_align(cursor_after_tensor_info, GGUF_DEFAULT_ALIGNMENT)


def tensor_data_base_is_aligned(offset: int, alignment: int) -> bool:
    if validate_alignment(alignment) != GGUF_TDBASE_OK:
        return False
    return (offset & (alignment - 1)) == 0


def tensor_data_base_offset(tensor_data_base: int, tensor_rel_offset: int, alignment: int):
    if validate_alignment(alignment) != GGUF_TDBASE_OK:
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT, 0
    if not tensor_data_base_is_aligned(tensor_data_base, alignment):
        return GGUF_TDBASE_ERR_MISALIGNED_BASE, 0
    if not tensor_data_base_is_aligned(tensor_rel_offset, alignment):
        return GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET, 0
    if tensor_data_base > U64_MAX - tensor_rel_offset:
        return GGUF_TDBASE_ERR_OVERFLOW, 0
    return GGUF_TDBASE_OK, tensor_data_base + tensor_rel_offset


def tensor_data_base_offset_default(tensor_data_base: int, tensor_rel_offset: int):
    return tensor_data_base_offset(tensor_data_base, tensor_rel_offset, GGUF_DEFAULT_ALIGNMENT)


def tensor_resolve_range(
    tensor_data_base: int,
    tensor_rel_offset: int,
    tensor_nbytes: int,
    gguf_file_nbytes: int,
    alignment: int,
):
    err, abs_start = tensor_data_base_offset(tensor_data_base, tensor_rel_offset, alignment)
    if err != GGUF_TDBASE_OK:
        return err, 0, 0

    if abs_start > U64_MAX - tensor_nbytes:
        return GGUF_TDBASE_ERR_OVERFLOW, 0, 0

    abs_end = abs_start + tensor_nbytes
    if abs_end > gguf_file_nbytes:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0

    return GGUF_TDBASE_OK, abs_start, abs_end


def tensor_resolve_range_default(
    tensor_data_base: int,
    tensor_rel_offset: int,
    tensor_nbytes: int,
    gguf_file_nbytes: int,
):
    return tensor_resolve_range(
        tensor_data_base,
        tensor_rel_offset,
        tensor_nbytes,
        gguf_file_nbytes,
        GGUF_DEFAULT_ALIGNMENT,
    )


def validate_tensor_ranges(
    tensor_data_base: int,
    alignment: int,
    gguf_file_nbytes: int,
    tensor_rel_offsets: list[int],
    tensor_nbytes: list[int],
):
    if len(tensor_rel_offsets) != len(tensor_nbytes):
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    count = len(tensor_rel_offsets)

    for i in range(count):
        err, _, _ = tensor_resolve_range(
            tensor_data_base,
            tensor_rel_offsets[i],
            tensor_nbytes[i],
            gguf_file_nbytes,
            alignment,
        )
        if err != GGUF_TDBASE_OK:
            return err, i

    for i in range(count):
        err, start_i, end_i = tensor_resolve_range(
            tensor_data_base,
            tensor_rel_offsets[i],
            tensor_nbytes[i],
            gguf_file_nbytes,
            alignment,
        )
        assert err == GGUF_TDBASE_OK

        for j in range(i + 1, count):
            err, start_j, end_j = tensor_resolve_range(
                tensor_data_base,
                tensor_rel_offsets[j],
                tensor_nbytes[j],
                gguf_file_nbytes,
                alignment,
            )
            assert err == GGUF_TDBASE_OK

            if (start_i < end_j) and (start_j < end_i):
                return GGUF_TDBASE_ERR_OVERLAP, j

    return GGUF_TDBASE_OK, 0


def _sift_down(rel_offsets: list[int], nbytes: list[int], root: int, end: int) -> None:
    def greater(i: int, j: int) -> bool:
        if rel_offsets[i] > rel_offsets[j]:
            return True
        if rel_offsets[i] < rel_offsets[j]:
            return False
        return nbytes[i] > nbytes[j]

    while True:
        child = root * 2 + 1
        if child > end:
            return

        swap_idx = root
        if greater(child, swap_idx):
            swap_idx = child
        if child + 1 <= end and greater(child + 1, swap_idx):
            swap_idx = child + 1

        if swap_idx == root:
            return

        rel_offsets[root], rel_offsets[swap_idx] = rel_offsets[swap_idx], rel_offsets[root]
        nbytes[root], nbytes[swap_idx] = nbytes[swap_idx], nbytes[root]
        root = swap_idx


def _heapsort_ranges_in_place(rel_offsets: list[int], nbytes: list[int]) -> None:
    count = len(rel_offsets)
    if count < 2:
        return

    start = ((count - 2) // 2) + 1
    while start > 0:
        start -= 1
        _sift_down(rel_offsets, nbytes, start, count - 1)

    end = count - 1
    while end > 0:
        rel_offsets[0], rel_offsets[end] = rel_offsets[end], rel_offsets[0]
        nbytes[0], nbytes[end] = nbytes[end], nbytes[0]
        end -= 1
        _sift_down(rel_offsets, nbytes, 0, end)


def validate_tensor_ranges_sorted(
    tensor_data_base: int,
    alignment: int,
    gguf_file_nbytes: int,
    tensor_rel_offsets: list[int],
    tensor_nbytes: list[int],
):
    if len(tensor_rel_offsets) != len(tensor_nbytes):
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    count = len(tensor_rel_offsets)

    for i in range(count):
        err, _, _ = tensor_resolve_range(
            tensor_data_base,
            tensor_rel_offsets[i],
            tensor_nbytes[i],
            gguf_file_nbytes,
            alignment,
        )
        if err != GGUF_TDBASE_OK:
            return err, i

    if count < 2:
        return GGUF_TDBASE_OK, 0

    _heapsort_ranges_in_place(tensor_rel_offsets, tensor_nbytes)

    err, _, max_end = tensor_resolve_range(
        tensor_data_base,
        tensor_rel_offsets[0],
        tensor_nbytes[0],
        gguf_file_nbytes,
        alignment,
    )
    assert err == GGUF_TDBASE_OK

    for i in range(1, count):
        err, start_i, end_i = tensor_resolve_range(
            tensor_data_base,
            tensor_rel_offsets[i],
            tensor_nbytes[i],
            gguf_file_nbytes,
            alignment,
        )
        assert err == GGUF_TDBASE_OK

        if start_i < max_end:
            return GGUF_TDBASE_ERR_OVERLAP, i

        if end_i > max_end:
            max_end = end_i

    return GGUF_TDBASE_OK, 0


def tensor_info_resolve_byte_spans(
    tensor_data_base: int,
    alignment: int,
    gguf_file_nbytes: int,
    tensor_rel_offsets: list[int],
    tensor_element_counts: list[int],
    tensor_ggml_types: list[int],
    out_tensor_nbytes: list[int],
):
    count = len(tensor_rel_offsets)

    if len(tensor_element_counts) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(tensor_ggml_types) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(out_tensor_nbytes) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    for i in range(count):
        err, nbytes = tensor_bytes_for_type(tensor_ggml_types[i], tensor_element_counts[i])
        if err != GGUF_TDBASE_OK:
            return err, i
        out_tensor_nbytes[i] = nbytes

    return validate_tensor_ranges_sorted(
        tensor_data_base=tensor_data_base,
        alignment=alignment,
        gguf_file_nbytes=gguf_file_nbytes,
        tensor_rel_offsets=tensor_rel_offsets,
        tensor_nbytes=out_tensor_nbytes,
    )


def tensor_info_resolve_abs_ranges(
    tensor_data_base: int,
    alignment: int,
    gguf_file_nbytes: int,
    tensor_rel_offsets: list[int],
    tensor_element_counts: list[int],
    tensor_ggml_types: list[int],
    out_tensor_nbytes: list[int],
    out_abs_starts: list[int],
    out_abs_ends: list[int],
):
    count = len(tensor_rel_offsets)

    if len(out_abs_starts) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(out_abs_ends) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    err, bad = tensor_info_resolve_byte_spans(
        tensor_data_base=tensor_data_base,
        alignment=alignment,
        gguf_file_nbytes=gguf_file_nbytes,
        tensor_rel_offsets=tensor_rel_offsets,
        tensor_element_counts=tensor_element_counts,
        tensor_ggml_types=tensor_ggml_types,
        out_tensor_nbytes=out_tensor_nbytes,
    )
    if err != GGUF_TDBASE_OK:
        return err, bad

    for i in range(count):
        err_off, abs_start = tensor_data_base_offset(
            tensor_data_base, tensor_rel_offsets[i], alignment
        )
        if err_off != GGUF_TDBASE_OK:
            return err_off, i

        nbytes = out_tensor_nbytes[i]
        if abs_start > U64_MAX - nbytes:
            return GGUF_TDBASE_ERR_OVERFLOW, i

        out_abs_starts[i] = abs_start
        out_abs_ends[i] = abs_start + nbytes

    return GGUF_TDBASE_OK, 0


def tensor_range_find_by_abs_offset(
    abs_offset: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
):
    count = len(tensor_abs_starts)
    if len(tensor_abs_ends) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if count == 0:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0

    lo = 0
    hi = count
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        start_mid = tensor_abs_starts[mid]
        end_mid = tensor_abs_ends[mid]

        if end_mid < start_mid:
            return GGUF_TDBASE_ERR_OVERFLOW, 0

        if abs_offset < start_mid:
            hi = mid
            continue

        if abs_offset >= end_mid:
            lo = mid + 1
            continue

        return GGUF_TDBASE_OK, mid

    return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0


def tensor_range_find_by_rel_offset(
    tensor_rel_offset: int,
    tensor_data_base: int,
    alignment: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
):
    err, abs_offset = tensor_data_base_offset(
        tensor_data_base=tensor_data_base,
        tensor_rel_offset=tensor_rel_offset,
        alignment=alignment,
    )
    if err != GGUF_TDBASE_OK:
        return err, 0

    return tensor_range_find_by_abs_offset(
        abs_offset=abs_offset,
        tensor_abs_starts=tensor_abs_starts,
        tensor_abs_ends=tensor_abs_ends,
    )


def tensor_range_find_by_rel_offset_default(
    tensor_rel_offset: int,
    tensor_data_base: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
):
    return tensor_range_find_by_rel_offset(
        tensor_rel_offset=tensor_rel_offset,
        tensor_data_base=tensor_data_base,
        alignment=GGUF_DEFAULT_ALIGNMENT,
        tensor_abs_starts=tensor_abs_starts,
        tensor_abs_ends=tensor_abs_ends,
    )


def tensor_range_find_index_by_abs_offset(
    abs_offset: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
    sorted_tensor_indices: list[int],
):
    count = len(tensor_abs_starts)
    if len(tensor_abs_ends) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(sorted_tensor_indices) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    err, sorted_pos = tensor_range_find_by_abs_offset(
        abs_offset=abs_offset,
        tensor_abs_starts=tensor_abs_starts,
        tensor_abs_ends=tensor_abs_ends,
    )
    if err != GGUF_TDBASE_OK:
        return err, 0

    if sorted_pos >= count:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0

    orig_idx = sorted_tensor_indices[sorted_pos]
    if orig_idx >= count:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0

    return GGUF_TDBASE_OK, orig_idx


def tensor_range_find_index_by_rel_offset(
    tensor_rel_offset: int,
    tensor_data_base: int,
    alignment: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
    sorted_tensor_indices: list[int],
):
    err, abs_offset = tensor_data_base_offset(
        tensor_data_base=tensor_data_base,
        tensor_rel_offset=tensor_rel_offset,
        alignment=alignment,
    )
    if err != GGUF_TDBASE_OK:
        return err, 0

    return tensor_range_find_index_by_abs_offset(
        abs_offset=abs_offset,
        tensor_abs_starts=tensor_abs_starts,
        tensor_abs_ends=tensor_abs_ends,
        sorted_tensor_indices=sorted_tensor_indices,
    )


def tensor_range_find_index_by_rel_offset_default(
    tensor_rel_offset: int,
    tensor_data_base: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
    sorted_tensor_indices: list[int],
):
    return tensor_range_find_index_by_rel_offset(
        tensor_rel_offset=tensor_rel_offset,
        tensor_data_base=tensor_data_base,
        alignment=GGUF_DEFAULT_ALIGNMENT,
        tensor_abs_starts=tensor_abs_starts,
        tensor_abs_ends=tensor_abs_ends,
        sorted_tensor_indices=sorted_tensor_indices,
    )


def validate_sorted_tensor_index(
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
    sorted_tensor_indices: list[int],
):
    count = len(tensor_abs_starts)
    if len(tensor_abs_ends) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(sorted_tensor_indices) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    for i in range(count):
        start_i = tensor_abs_starts[i]
        end_i = tensor_abs_ends[i]
        idx_i = sorted_tensor_indices[i]

        if end_i < start_i:
            return GGUF_TDBASE_ERR_OVERFLOW, i

        if i > 0:
            prev_start = tensor_abs_starts[i - 1]
            prev_end = tensor_abs_ends[i - 1]

            if start_i < prev_start:
                return GGUF_TDBASE_ERR_OVERLAP, i
            if start_i == prev_start and end_i < prev_end:
                return GGUF_TDBASE_ERR_OVERLAP, i
            if start_i < prev_end:
                return GGUF_TDBASE_ERR_OVERLAP, i

        if idx_i >= count:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

        for j in range(i):
            if sorted_tensor_indices[j] == idx_i:
                return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

    return GGUF_TDBASE_OK, 0


def build_tensor_sorted_position_map(
    sorted_tensor_indices: list[int],
    out_sorted_position_by_tensor: list[int],
):
    count = len(sorted_tensor_indices)
    if len(out_sorted_position_by_tensor) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    for i in range(count):
        out_sorted_position_by_tensor[i] = count

    for i in range(count):
        tensor_index = sorted_tensor_indices[i]

        if tensor_index >= count:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

        if out_sorted_position_by_tensor[tensor_index] != count:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

        out_sorted_position_by_tensor[tensor_index] = i

    for i in range(count):
        if out_sorted_position_by_tensor[i] == count:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

    return GGUF_TDBASE_OK, 0


def validate_sorted_position_map(
    sorted_tensor_indices: list[int],
    sorted_position_by_tensor: list[int],
):
    count = len(sorted_tensor_indices)
    if len(sorted_position_by_tensor) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    for i in range(count):
        sorted_pos = sorted_position_by_tensor[i]

        if sorted_pos >= count:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

        for j in range(i):
            if sorted_position_by_tensor[j] == sorted_pos:
                return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

        if sorted_tensor_indices[sorted_pos] != i:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

    for i in range(count):
        orig_idx = sorted_tensor_indices[i]

        if orig_idx >= count:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

        if sorted_position_by_tensor[orig_idx] != i:
            return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, i

    return GGUF_TDBASE_OK, 0


def tensor_index_to_abs_range(
    tensor_index: int,
    tensor_abs_starts: list[int],
    tensor_abs_ends: list[int],
    sorted_position_by_tensor: list[int],
):
    count = len(tensor_abs_starts)
    if len(tensor_abs_ends) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0, 0
    if len(sorted_position_by_tensor) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0, 0

    if tensor_index >= count:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0

    sorted_pos = sorted_position_by_tensor[tensor_index]
    if sorted_pos >= count:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0

    abs_start = tensor_abs_starts[sorted_pos]
    abs_end = tensor_abs_ends[sorted_pos]

    if abs_end < abs_start:
        return GGUF_TDBASE_ERR_OVERFLOW, 0, 0

    return GGUF_TDBASE_OK, abs_start, abs_end


def _abs_range_greater(
    starts: list[int],
    ends: list[int],
    indices: list[int],
    i: int,
    j: int,
) -> bool:
    if starts[i] > starts[j]:
        return True
    if starts[i] < starts[j]:
        return False
    if ends[i] > ends[j]:
        return True
    if ends[i] < ends[j]:
        return False
    return indices[i] > indices[j]


def _abs_range_sift_down(
    starts: list[int],
    ends: list[int],
    indices: list[int],
    root: int,
    end: int,
) -> None:
    while True:
        child = root * 2 + 1
        if child > end:
            return

        swap_idx = root
        if _abs_range_greater(starts, ends, indices, child, swap_idx):
            swap_idx = child
        if child + 1 <= end and _abs_range_greater(starts, ends, indices, child + 1, swap_idx):
            swap_idx = child + 1

        if swap_idx == root:
            return

        starts[root], starts[swap_idx] = starts[swap_idx], starts[root]
        ends[root], ends[swap_idx] = ends[swap_idx], ends[root]
        indices[root], indices[swap_idx] = indices[swap_idx], indices[root]
        root = swap_idx


def _sort_abs_ranges_with_indices(
    starts: list[int],
    ends: list[int],
    indices: list[int],
) -> None:
    count = len(starts)
    if count < 2:
        return

    start = ((count - 2) // 2) + 1
    while start > 0:
        start -= 1
        _abs_range_sift_down(starts, ends, indices, start, count - 1)

    end = count - 1
    while end > 0:
        starts[0], starts[end] = starts[end], starts[0]
        ends[0], ends[end] = ends[end], ends[0]
        indices[0], indices[end] = indices[end], indices[0]
        end -= 1
        _abs_range_sift_down(starts, ends, indices, 0, end)


def tensor_info_build_offset_index(
    tensor_data_base: int,
    alignment: int,
    gguf_file_nbytes: int,
    tensor_rel_offsets: list[int],
    tensor_element_counts: list[int],
    tensor_ggml_types: list[int],
    out_abs_starts: list[int],
    out_abs_ends: list[int],
    out_sorted_tensor_indices: list[int],
):
    count = len(tensor_rel_offsets)

    if len(tensor_element_counts) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(tensor_ggml_types) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(out_abs_starts) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(out_abs_ends) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0
    if len(out_sorted_tensor_indices) != count:
        return GGUF_TDBASE_ERR_NULL_PTR, 0

    for i in range(count):
        err, nbytes = tensor_bytes_for_type(tensor_ggml_types[i], tensor_element_counts[i])
        if err != GGUF_TDBASE_OK:
            return err, i

        err, abs_start, abs_end = tensor_resolve_range(
            tensor_data_base=tensor_data_base,
            tensor_rel_offset=tensor_rel_offsets[i],
            tensor_nbytes=nbytes,
            gguf_file_nbytes=gguf_file_nbytes,
            alignment=alignment,
        )
        if err != GGUF_TDBASE_OK:
            return err, i

        out_abs_starts[i] = abs_start
        out_abs_ends[i] = abs_end
        out_sorted_tensor_indices[i] = i

    if count < 2:
        return GGUF_TDBASE_OK, 0

    _sort_abs_ranges_with_indices(out_abs_starts, out_abs_ends, out_sorted_tensor_indices)

    max_end = out_abs_ends[0]
    for i in range(1, count):
        if out_abs_ends[i] < out_abs_starts[i]:
            return GGUF_TDBASE_ERR_OVERFLOW, out_sorted_tensor_indices[i]
        if out_abs_starts[i] < max_end:
            return GGUF_TDBASE_ERR_OVERLAP, out_sorted_tensor_indices[i]
        if out_abs_ends[i] > max_end:
            max_end = out_abs_ends[i]

    return GGUF_TDBASE_OK, 0




def test_tensor_bytes_for_type_scalars() -> None:
    assert tensor_bytes_for_type(GGML_TYPE_F32, 0) == (GGUF_TDBASE_OK, 0)
    assert tensor_bytes_for_type(GGML_TYPE_F32, 7) == (GGUF_TDBASE_OK, 28)
    assert tensor_bytes_for_type(GGML_TYPE_F16, 7) == (GGUF_TDBASE_OK, 14)


def test_tensor_bytes_for_type_quantized_exact_blocks() -> None:
    assert tensor_bytes_for_type(GGML_TYPE_Q4_0, 32) == (GGUF_TDBASE_OK, 18)
    assert tensor_bytes_for_type(GGML_TYPE_Q4_0, 64) == (GGUF_TDBASE_OK, 36)
    assert tensor_bytes_for_type(GGML_TYPE_Q8_0, 32) == (GGUF_TDBASE_OK, 34)
    assert tensor_bytes_for_type(GGML_TYPE_Q8_0, 96) == (GGUF_TDBASE_OK, 102)


def test_tensor_bytes_for_type_reject_non_multiple_block() -> None:
    assert tensor_bytes_for_type(GGML_TYPE_Q4_0, 31) == (GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0)
    assert tensor_bytes_for_type(GGML_TYPE_Q8_0, 33) == (GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0)


def test_tensor_bytes_for_type_reject_unknown_type() -> None:
    assert tensor_bytes_for_type(999, 32) == (GGUF_TDBASE_ERR_BAD_TYPE, 0)


def test_tensor_bytes_for_type_overflow_guard() -> None:
    q8_block = 32
    q8_bytes = 34
    max_blocks = U64_MAX // q8_bytes
    ok_elements = max_blocks * q8_block
    err, nbytes = tensor_bytes_for_type(GGML_TYPE_Q8_0, ok_elements)
    assert err == GGUF_TDBASE_OK
    assert nbytes == max_blocks * q8_bytes

    overflow_elements = (max_blocks + 1) * q8_block
    err, _ = tensor_bytes_for_type(GGML_TYPE_Q8_0, overflow_elements)
    assert err == GGUF_TDBASE_ERR_OVERFLOW

def test_default_alignment_examples() -> None:
    assert tensor_data_base_align_default(0) == (GGUF_TDBASE_OK, 0)
    assert tensor_data_base_align_default(1) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(31) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(32) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(33) == (GGUF_TDBASE_OK, 64)


def test_tensor_type_block_size_known_types() -> None:
    assert tensor_type_block_size(GGML_TYPE_F32) == (GGUF_TDBASE_OK, 1)
    assert tensor_type_block_size(GGML_TYPE_F16) == (GGUF_TDBASE_OK, 1)
    assert tensor_type_block_size(GGML_TYPE_Q4_0) == (GGUF_TDBASE_OK, 32)
    assert tensor_type_block_size(GGML_TYPE_Q8_0) == (GGUF_TDBASE_OK, 32)


def test_tensor_type_block_bytes_known_types() -> None:
    assert tensor_type_block_bytes(GGML_TYPE_F32) == (GGUF_TDBASE_OK, 4)
    assert tensor_type_block_bytes(GGML_TYPE_F16) == (GGUF_TDBASE_OK, 2)
    assert tensor_type_block_bytes(GGML_TYPE_Q4_0) == (GGUF_TDBASE_OK, 18)
    assert tensor_type_block_bytes(GGML_TYPE_Q8_0) == (GGUF_TDBASE_OK, 34)


def test_tensor_type_helpers_reject_unknown_type() -> None:
    assert tensor_type_block_size(999) == (GGUF_TDBASE_ERR_BAD_TYPE, 0)
    assert tensor_type_block_bytes(999) == (GGUF_TDBASE_ERR_BAD_TYPE, 0)


def test_custom_power_of_two_alignment() -> None:
    err, base = tensor_data_base_align(0x1234, 64)
    assert err == GGUF_TDBASE_OK
    assert base == 0x1240
    assert tensor_data_base_is_aligned(base, 64)


def test_reject_non_power_of_two_alignment() -> None:
    assert tensor_data_base_align(100, 0)[0] == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert tensor_data_base_align(100, 3)[0] == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert tensor_data_base_align(100, 24)[0] == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert tensor_data_base_is_aligned(128, 24) is False


def test_overflow_guard_near_u64_max() -> None:
    err, _ = tensor_data_base_align(U64_MAX - 7, 16)
    assert err == GGUF_TDBASE_ERR_OVERFLOW

    err, base = tensor_data_base_align(U64_MAX - 15, 16)
    assert err == GGUF_TDBASE_OK
    assert base == U64_MAX - 15


def test_random_alignment_matches_reference_formula() -> None:
    rng = random.Random(760031)

    for _ in range(1000):
        alignment = 1 << rng.randint(0, 20)
        cursor = rng.randrange(0, 1 << 56)

        err, got = tensor_data_base_align(cursor, alignment)
        assert err == GGUF_TDBASE_OK

        reference = ((cursor + (alignment - 1)) // alignment) * alignment
        assert got == reference
        assert got >= cursor
        assert tensor_data_base_is_aligned(got, alignment)


def test_tensor_data_base_offset_happy_path() -> None:
    err, abs_off = tensor_data_base_offset(0x2000, 0x80, 32)
    assert err == GGUF_TDBASE_OK
    assert abs_off == 0x2080



def test_tensor_data_base_offset_default_alignment() -> None:
    err, abs_off = tensor_data_base_offset_default(0x100, 0x40)
    assert err == GGUF_TDBASE_OK
    assert abs_off == 0x140



def test_tensor_data_base_offset_reject_bad_alignment() -> None:
    err, _ = tensor_data_base_offset(0x1000, 0x40, 24)
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT



def test_tensor_data_base_offset_reject_misaligned_base() -> None:
    err, _ = tensor_data_base_offset(0x1010, 0x40, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_BASE



def test_tensor_data_base_offset_reject_misaligned_rel() -> None:
    err, _ = tensor_data_base_offset(0x1000, 0x48, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET



def test_tensor_data_base_offset_overflow_guard() -> None:
    err, _ = tensor_data_base_offset(U64_MAX - 31, 32, 32)
    assert err == GGUF_TDBASE_ERR_OVERFLOW



def test_tensor_data_base_offset_random_aligned_cases() -> None:
    rng = random.Random(761113)

    for _ in range(1000):
        alignment = 1 << rng.randint(0, 15)
        base_units = rng.randrange(0, 1 << 24)
        rel_units = rng.randrange(0, 1 << 20)
        base = base_units * alignment
        rel = rel_units * alignment

        err, got = tensor_data_base_offset(base, rel, alignment)
        assert err == GGUF_TDBASE_OK
        assert got == base + rel
        assert tensor_data_base_is_aligned(got, alignment)


def test_tensor_resolve_range_happy_path() -> None:
    err, abs_start, abs_end = tensor_resolve_range(0x2000, 0x80, 0x120, 0x3000, 32)
    assert err == GGUF_TDBASE_OK
    assert abs_start == 0x2080
    assert abs_end == 0x21A0


def test_tensor_resolve_range_default_alignment() -> None:
    err, abs_start, abs_end = tensor_resolve_range_default(0x100, 0x40, 0x10, 0x1000)
    assert err == GGUF_TDBASE_OK
    assert abs_start == 0x140
    assert abs_end == 0x150


def test_tensor_resolve_range_reject_bad_alignment() -> None:
    err, _, _ = tensor_resolve_range(0x100, 0x20, 0x10, 0x1000, 24)
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT


def test_tensor_resolve_range_reject_misaligned_base() -> None:
    err, _, _ = tensor_resolve_range(0x110, 0x20, 0x10, 0x1000, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_BASE


def test_tensor_resolve_range_reject_misaligned_offset() -> None:
    err, _, _ = tensor_resolve_range(0x100, 0x18, 0x10, 0x1000, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET


def test_tensor_resolve_range_overflow_guard() -> None:
    err, _, _ = tensor_resolve_range(U64_MAX - 0x1F, 0x20, 0x40, U64_MAX, 32)
    assert err == GGUF_TDBASE_ERR_OVERFLOW


def test_tensor_resolve_range_out_of_bounds() -> None:
    err, _, _ = tensor_resolve_range(0x200, 0x40, 0xC0, 0x2F0, 32)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS


def test_tensor_resolve_range_allows_exact_eof() -> None:
    err, abs_start, abs_end = tensor_resolve_range(0x400, 0x80, 0x100, 0x580, 32)
    assert err == GGUF_TDBASE_OK
    assert abs_start == 0x480
    assert abs_end == 0x580


def test_tensor_resolve_range_random_valid_cases() -> None:
    rng = random.Random(761499)

    for _ in range(1000):
        alignment = 1 << rng.randint(0, 15)
        base_units = rng.randrange(0, 1 << 20)
        rel_units = rng.randrange(0, 1 << 20)
        tensor_nbytes = rng.randrange(0, 1 << 16)

        base = base_units * alignment
        rel = rel_units * alignment
        start = base + rel
        if start > U64_MAX - tensor_nbytes:
            continue
        end = start + tensor_nbytes

        file_pad = rng.randrange(0, 1 << 12)
        file_nbytes = end + file_pad

        err, got_start, got_end = tensor_resolve_range(
            base,
            rel,
            tensor_nbytes,
            file_nbytes,
            alignment,
        )
        assert err == GGUF_TDBASE_OK
        assert got_start == start
        assert got_end == end


def test_validate_tensor_ranges_non_overlapping_happy_path() -> None:
    err, bad = validate_tensor_ranges(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=[0x00, 0x80, 0x100, 0x200],
        tensor_nbytes=[0x40, 0x40, 0x80, 0x20],
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_tensor_ranges_detect_overlap() -> None:
    err, bad = validate_tensor_ranges(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=[0x00, 0x20, 0x200],
        tensor_nbytes=[0x40, 0x40, 0x20],
    )
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 1


def test_validate_tensor_ranges_propagates_bad_alignment() -> None:
    err, bad = validate_tensor_ranges(
        tensor_data_base=0x1000,
        alignment=24,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=[0x00, 0x20],
        tensor_nbytes=[0x20, 0x20],
    )
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert bad == 0


def test_validate_tensor_ranges_propagates_out_of_bounds_with_index() -> None:
    err, bad = validate_tensor_ranges(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x1100,
        tensor_rel_offsets=[0x00, 0x80],
        tensor_nbytes=[0x20, 0x120],
    )
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 1


def test_validate_tensor_ranges_random_non_overlapping() -> None:
    rng = random.Random(761991)

    for _ in range(300):
        alignment = 1 << rng.randint(0, 10)
        count = rng.randint(1, 12)
        cursor = 0

        rel_offsets = []
        nbytes = []
        for _ in range(count):
            gap_units = rng.randint(0, 8)
            gap = gap_units * alignment
            cursor += gap

            span_units = rng.randint(0, 6)
            span = span_units * alignment

            rel_offsets.append(cursor)
            nbytes.append(span)
            cursor += span

        base_units = rng.randint(0, 1024)
        base = base_units * alignment
        file_nbytes = base + cursor + (rng.randint(0, 16) * alignment)

        err, bad = validate_tensor_ranges(
            tensor_data_base=base,
            alignment=alignment,
            gguf_file_nbytes=file_nbytes,
            tensor_rel_offsets=rel_offsets,
            tensor_nbytes=nbytes,
        )
        assert err == GGUF_TDBASE_OK
        assert bad == 0


def test_validate_tensor_ranges_random_overlap_cases() -> None:
    rng = random.Random(762007)

    for _ in range(200):
        alignment = 1 << rng.randint(0, 10)
        base = rng.randint(0, 1024) * alignment

        a_start = rng.randint(0, 128) * alignment
        a_len = max(alignment, rng.randint(1, 8) * alignment)
        b_start = a_start + rng.randint(0, (a_len // alignment) - 1) * alignment
        b_len = max(alignment, rng.randint(1, 8) * alignment)

        rel_offsets = [a_start, b_start]
        nbytes = [a_len, b_len]
        max_end = max(a_start + a_len, b_start + b_len)
        file_nbytes = base + max_end + alignment

        err, bad = validate_tensor_ranges(
            tensor_data_base=base,
            alignment=alignment,
            gguf_file_nbytes=file_nbytes,
            tensor_rel_offsets=rel_offsets,
            tensor_nbytes=nbytes,
        )
        assert err == GGUF_TDBASE_ERR_OVERLAP
        assert bad == 1


def test_validate_tensor_ranges_sorted_non_overlapping_happy_path() -> None:
    rel_offsets = [0x200, 0x00, 0x100, 0x80]
    nbytes = [0x20, 0x40, 0x40, 0x40]

    err, bad = validate_tensor_ranges_sorted(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_nbytes=nbytes,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0
    assert rel_offsets == [0x00, 0x80, 0x100, 0x200]
    assert nbytes == [0x40, 0x40, 0x40, 0x20]


def test_validate_tensor_ranges_sorted_tie_breaks_equal_offsets_by_size() -> None:
    rel_offsets = [0x80, 0x80, 0x80]
    nbytes = [0x20, 0x00, 0x08]

    err, bad = validate_tensor_ranges_sorted(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_nbytes=nbytes,
    )
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 2
    assert rel_offsets == [0x80, 0x80, 0x80]
    assert nbytes == [0x00, 0x08, 0x20]


def test_validate_tensor_ranges_sorted_detect_overlap() -> None:
    rel_offsets = [0x100, 0x00, 0x20]
    nbytes = [0x20, 0x40, 0x40]

    err, bad = validate_tensor_ranges_sorted(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_nbytes=nbytes,
    )
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 1
    assert rel_offsets == [0x00, 0x20, 0x100]
    assert nbytes == [0x40, 0x40, 0x20]


def test_validate_tensor_ranges_sorted_propagates_bad_alignment_without_sort() -> None:
    rel_offsets = [0x80, 0x00]
    nbytes = [0x20, 0x20]

    err, bad = validate_tensor_ranges_sorted(
        tensor_data_base=0x1000,
        alignment=24,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_nbytes=nbytes,
    )
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert bad == 0
    assert rel_offsets == [0x80, 0x00]
    assert nbytes == [0x20, 0x20]


def test_validate_tensor_ranges_sorted_random_matches_quadratic_validator() -> None:
    rng = random.Random(762199)

    for _ in range(500):
        alignment = 1 << rng.randint(0, 10)
        count = rng.randint(1, 24)
        base = rng.randint(0, 1024) * alignment

        rel_offsets = []
        nbytes = []
        cursor = 0

        force_overlap = rng.random() < 0.5 and count >= 2
        overlap_i = rng.randint(0, count - 2) if force_overlap else -1

        for i in range(count):
            gap = rng.randint(0, 6) * alignment
            cursor += gap
            span = rng.randint(0, 6) * alignment

            start = cursor
            if force_overlap and i == overlap_i + 1:
                prev_start = rel_offsets[-1]
                prev_span = nbytes[-1]
                if prev_span == 0:
                    start = prev_start
                else:
                    start = prev_start + rng.randint(0, max(0, (prev_span // alignment) - 1)) * alignment

            rel_offsets.append(start)
            nbytes.append(span)
            cursor = max(cursor, start + span)

        file_nbytes = base + cursor + rng.randint(0, 8) * alignment + alignment

        rel_a = rel_offsets.copy()
        nbytes_a = nbytes.copy()
        err_quad, bad_quad = validate_tensor_ranges(
            tensor_data_base=base,
            alignment=alignment,
            gguf_file_nbytes=file_nbytes,
            tensor_rel_offsets=rel_a,
            tensor_nbytes=nbytes_a,
        )

        rel_b = rel_offsets.copy()
        nbytes_b = nbytes.copy()
        err_sorted, bad_sorted = validate_tensor_ranges_sorted(
            tensor_data_base=base,
            alignment=alignment,
            gguf_file_nbytes=file_nbytes,
            tensor_rel_offsets=rel_b,
            tensor_nbytes=nbytes_b,
        )

        assert err_sorted == err_quad
        if err_sorted == GGUF_TDBASE_ERR_OVERLAP:
            assert bad_sorted >= 1
        elif err_sorted == GGUF_TDBASE_OK:
            assert bad_sorted == 0
            assert rel_b == sorted(rel_offsets)


def test_tensor_info_resolve_byte_spans_happy_path() -> None:
    rel_offsets = [0x80, 0x00, 0x40]
    element_counts = [32, 8, 32]
    ggml_types = [GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_Q8_0]
    out_nbytes = [0, 0, 0]

    err, bad = tensor_info_resolve_byte_spans(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x4000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0
    assert rel_offsets == [0x00, 0x40, 0x80]
    assert out_nbytes == [32, 34, 18]


def test_tensor_info_resolve_byte_spans_reject_bad_block_multiple() -> None:
    rel_offsets = [0x00, 0x20]
    element_counts = [32, 31]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0]
    out_nbytes = [0, 0]

    err, bad = tensor_info_resolve_byte_spans(
        tensor_data_base=0x2000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
    )
    assert err == GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE
    assert bad == 1


def test_tensor_info_resolve_byte_spans_reject_unknown_type() -> None:
    rel_offsets = [0x00]
    element_counts = [32]
    ggml_types = [999]
    out_nbytes = [0]

    err, bad = tensor_info_resolve_byte_spans(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
    )
    assert err == GGUF_TDBASE_ERR_BAD_TYPE
    assert bad == 0


def test_tensor_info_resolve_byte_spans_detect_overlap_after_sizing() -> None:
    rel_offsets = [0x00, 0x20]
    element_counts = [16, 8]
    ggml_types = [GGML_TYPE_F32, GGML_TYPE_F32]
    out_nbytes = [0, 0]

    err, bad = tensor_info_resolve_byte_spans(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
    )
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 1


def test_tensor_info_resolve_abs_ranges_happy_path() -> None:
    rel_offsets = [0x80, 0x00, 0x40]
    element_counts = [32, 8, 32]
    ggml_types = [GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_Q8_0]
    out_nbytes = [0, 0, 0]
    out_abs_starts = [0, 0, 0]
    out_abs_ends = [0, 0, 0]

    err, bad = tensor_info_resolve_abs_ranges(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x4000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
        out_abs_starts=out_abs_starts,
        out_abs_ends=out_abs_ends,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0
    assert rel_offsets == [0x00, 0x40, 0x80]
    assert out_nbytes == [32, 34, 18]
    assert out_abs_starts == [0x1000, 0x1040, 0x1080]
    assert out_abs_ends == [0x1020, 0x1062, 0x1092]


def test_tensor_info_resolve_abs_ranges_propagates_bad_block_multiple() -> None:
    rel_offsets = [0x00, 0x20]
    element_counts = [32, 31]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0]
    out_nbytes = [0, 0]
    out_abs_starts = [0, 0]
    out_abs_ends = [0, 0]

    err, bad = tensor_info_resolve_abs_ranges(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x5000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
        out_abs_starts=out_abs_starts,
        out_abs_ends=out_abs_ends,
    )
    assert err == GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE
    assert bad == 1


def test_tensor_info_resolve_abs_ranges_reject_null_outputs() -> None:
    rel_offsets = [0x00]
    element_counts = [8]
    ggml_types = [GGML_TYPE_F32]
    out_nbytes = [0]

    err, bad = tensor_info_resolve_abs_ranges(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=element_counts,
        tensor_ggml_types=ggml_types,
        out_tensor_nbytes=out_nbytes,
        out_abs_starts=[],
        out_abs_ends=[0],
    )
    assert err == GGUF_TDBASE_ERR_NULL_PTR
    assert bad == 0


def test_tensor_range_find_by_abs_offset_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]

    assert tensor_range_find_by_abs_offset(0x1000, starts, ends) == (GGUF_TDBASE_OK, 0)
    assert tensor_range_find_by_abs_offset(0x101F, starts, ends) == (GGUF_TDBASE_OK, 0)
    assert tensor_range_find_by_abs_offset(0x1040, starts, ends) == (GGUF_TDBASE_OK, 1)
    assert tensor_range_find_by_abs_offset(0x1061, starts, ends) == (GGUF_TDBASE_OK, 1)
    assert tensor_range_find_by_abs_offset(0x1080, starts, ends) == (GGUF_TDBASE_OK, 2)


def test_tensor_range_find_by_abs_offset_gap_and_end_exclusive() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]

    assert tensor_range_find_by_abs_offset(0x1020, starts, ends)[0] == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert tensor_range_find_by_abs_offset(0x107F, starts, ends)[0] == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert tensor_range_find_by_abs_offset(0x1092, starts, ends)[0] == GGUF_TDBASE_ERR_OUT_OF_BOUNDS


def test_tensor_range_find_by_abs_offset_rejects_malformed_range() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1030]

    err, idx = tensor_range_find_by_abs_offset(0x1040, starts, ends)
    assert err == GGUF_TDBASE_ERR_OVERFLOW
    assert idx == 0


def test_tensor_range_find_by_abs_offset_empty_table() -> None:
    err, idx = tensor_range_find_by_abs_offset(0x1000, [], [])
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert idx == 0


def test_tensor_range_find_by_rel_offset_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]

    assert tensor_range_find_by_rel_offset(0x00, 0x1000, 32, starts, ends) == (GGUF_TDBASE_OK, 0)
    assert tensor_range_find_by_rel_offset(0x40, 0x1000, 32, starts, ends) == (GGUF_TDBASE_OK, 1)
    assert tensor_range_find_by_rel_offset(0x80, 0x1000, 32, starts, ends) == (GGUF_TDBASE_OK, 2)


def test_tensor_range_find_by_rel_offset_propagates_alignment_error() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]

    err, idx = tensor_range_find_by_rel_offset(0x40, 0x1000, 24, starts, ends)
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert idx == 0


def test_tensor_range_find_by_rel_offset_propagates_misaligned_rel() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]

    err, idx = tensor_range_find_by_rel_offset(0x48, 0x1000, 32, starts, ends)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET
    assert idx == 0


def test_tensor_range_find_by_rel_offset_gap_and_out_of_bounds() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]

    err, idx = tensor_range_find_by_rel_offset(0x20, 0x1000, 32, starts, ends)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert idx == 0

    err, idx = tensor_range_find_by_rel_offset(0xA0, 0x1000, 32, starts, ends)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert idx == 0


def test_tensor_range_find_by_rel_offset_default_alignment_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]

    assert tensor_range_find_by_rel_offset_default(0x00, 0x1000, starts, ends) == (GGUF_TDBASE_OK, 0)
    assert tensor_range_find_by_rel_offset_default(0x40, 0x1000, starts, ends) == (GGUF_TDBASE_OK, 1)
    assert tensor_range_find_by_rel_offset_default(0x80, 0x1000, starts, ends) == (GGUF_TDBASE_OK, 2)


def test_tensor_range_find_by_rel_offset_default_alignment_propagates_error() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]

    err, idx = tensor_range_find_by_rel_offset_default(0x48, 0x1000, starts, ends)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET
    assert idx == 0


def test_tensor_range_find_index_by_abs_offset_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]
    sorted_idx = [2, 0, 1]

    assert tensor_range_find_index_by_abs_offset(0x1000, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        2,
    )
    assert tensor_range_find_index_by_abs_offset(0x1050, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        0,
    )
    assert tensor_range_find_index_by_abs_offset(0x1085, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        1,
    )


def test_tensor_range_find_index_by_abs_offset_rejects_bad_index_table() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]
    sorted_idx = [0, 9]

    err, idx = tensor_range_find_index_by_abs_offset(0x1050, starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert idx == 0


def test_tensor_range_find_index_by_abs_offset_gap_propagates_miss() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]
    sorted_idx = [1, 0]

    err, idx = tensor_range_find_index_by_abs_offset(0x1030, starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert idx == 0


def test_tensor_range_find_index_by_rel_offset_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]
    sorted_idx = [2, 0, 1]

    assert tensor_range_find_index_by_rel_offset(0x00, 0x1000, 32, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        2,
    )
    assert tensor_range_find_index_by_rel_offset(0x40, 0x1000, 32, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        0,
    )
    assert tensor_range_find_index_by_rel_offset(0x80, 0x1000, 32, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        1,
    )


def test_tensor_range_find_index_by_rel_offset_propagates_alignment_error() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]
    sorted_idx = [0, 1]

    err, idx = tensor_range_find_index_by_rel_offset(0x40, 0x1000, 24, starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert idx == 0


def test_tensor_range_find_index_by_rel_offset_with_built_index() -> None:
    rel_offsets = [0xC0, 0x00, 0x40]
    elem_counts = [32, 64, 32]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16]
    starts = [0, 0, 0]
    ends = [0, 0, 0]
    sorted_idx = [0, 0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=elem_counts,
        tensor_ggml_types=ggml_types,
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    assert tensor_range_find_index_by_rel_offset(0x00, 0x1000, 32, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        1,
    )
    assert tensor_range_find_index_by_rel_offset(0x40, 0x1000, 32, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        2,
    )
    assert tensor_range_find_index_by_rel_offset(0xC0, 0x1000, 32, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        0,
    )


def test_tensor_range_find_index_by_rel_offset_default_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]
    sorted_idx = [2, 0, 1]

    assert tensor_range_find_index_by_rel_offset_default(0x00, 0x1000, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        2,
    )
    assert tensor_range_find_index_by_rel_offset_default(0x40, 0x1000, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        0,
    )
    assert tensor_range_find_index_by_rel_offset_default(0x80, 0x1000, starts, ends, sorted_idx) == (
        GGUF_TDBASE_OK,
        1,
    )


def test_tensor_range_find_index_by_rel_offset_default_propagates_error() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1020, 0x1060]
    sorted_idx = [0, 1]

    err, idx = tensor_range_find_index_by_rel_offset_default(0x48, 0x1000, starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET
    assert idx == 0


def test_tensor_info_build_offset_index_happy_path() -> None:
    rel_offsets = [0xC0, 0x00, 0x40]
    elem_counts = [32, 64, 32]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16]
    starts = [0, 0, 0]
    ends = [0, 0, 0]
    sorted_idx = [0, 0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=elem_counts,
        tensor_ggml_types=ggml_types,
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0
    assert starts == [0x1000, 0x1040, 0x10C0]
    assert ends == [0x1024, 0x1080, 0x10E2]
    assert sorted_idx == [1, 2, 0]


def test_tensor_info_build_offset_index_propagates_bad_block_multiple() -> None:
    starts = [0, 0]
    ends = [0, 0]
    sorted_idx = [0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x3000,
        tensor_rel_offsets=[0x00, 0x40],
        tensor_element_counts=[31, 32],
        tensor_ggml_types=[GGML_TYPE_Q4_0, GGML_TYPE_Q8_0],
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE
    assert bad == 0


def test_tensor_info_build_offset_index_detects_overlap() -> None:
    starts = [0, 0]
    ends = [0, 0]
    sorted_idx = [0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x3000,
        tensor_rel_offsets=[0x00, 0x20],
        tensor_element_counts=[32, 32],
        tensor_ggml_types=[GGML_TYPE_Q8_0, GGML_TYPE_Q8_0],
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 1


def test_tensor_info_build_offset_index_random_non_overlapping() -> None:
    rng = random.Random(764021)

    for _ in range(200):
        count = rng.randint(1, 12)
        rel_offsets: list[int] = []
        elem_counts: list[int] = []
        ggml_types: list[int] = []

        cursor = 0
        for _ in range(count):
            gap_units = rng.randint(0, 4)
            cursor += gap_units * 32

            ggml_type = rng.choice([GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32])
            if ggml_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q8_0):
                blocks = rng.randint(1, 4)
                elem_count = blocks * 32
            else:
                elem_count = rng.randint(1, 128)

            err, nbytes = tensor_bytes_for_type(ggml_type, elem_count)
            assert err == GGUF_TDBASE_OK

            rel_offsets.append(cursor)
            elem_counts.append(elem_count)
            ggml_types.append(ggml_type)

            span_units = (nbytes + 31) // 32
            cursor += span_units * 32

        starts = [0] * count
        ends = [0] * count
        sorted_idx = [0] * count
        file_nbytes = 0x1000 + cursor + 0x100

        err, bad = tensor_info_build_offset_index(
            tensor_data_base=0x1000,
            alignment=32,
            gguf_file_nbytes=file_nbytes,
            tensor_rel_offsets=rel_offsets,
            tensor_element_counts=elem_counts,
            tensor_ggml_types=ggml_types,
            out_abs_starts=starts,
            out_abs_ends=ends,
            out_sorted_tensor_indices=sorted_idx,
        )
        assert err == GGUF_TDBASE_OK
        assert bad == 0
        assert starts == sorted(starts)
        for i in range(1, count):
            assert starts[i] >= ends[i - 1]
        assert sorted(sorted_idx) == list(range(count))


def test_validate_sorted_tensor_index_happy_path() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1062, 0x1092]
    sorted_idx = [2, 0, 1]

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_sorted_tensor_index_empty_ok() -> None:
    err, bad = validate_sorted_tensor_index([], [], [])
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_sorted_tensor_index_detects_overflow_range() -> None:
    starts = [0x1000, 0x1080]
    ends = [0x1020, 0x107F]
    sorted_idx = [0, 1]

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OVERFLOW
    assert bad == 1


def test_validate_sorted_tensor_index_detects_unsorted_starts() -> None:
    starts = [0x1000, 0x1100, 0x1080]
    ends = [0x1020, 0x1120, 0x10A0]
    sorted_idx = [0, 1, 2]

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 2


def test_validate_sorted_tensor_index_detects_overlap() -> None:
    starts = [0x1000, 0x1040]
    ends = [0x1060, 0x1080]
    sorted_idx = [0, 1]

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OVERLAP
    assert bad == 1


def test_validate_sorted_tensor_index_rejects_bad_mapped_index() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1060, 0x10A0]
    sorted_idx = [1, 3, 0]

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 1


def test_validate_sorted_tensor_index_rejects_duplicate_mapped_index() -> None:
    starts = [0x1000, 0x1040, 0x1080]
    ends = [0x1020, 0x1060, 0x10A0]
    sorted_idx = [1, 0, 1]

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 2


def test_validate_sorted_tensor_index_with_built_index() -> None:
    rel_offsets = [0xC0, 0x00, 0x40]
    elem_counts = [32, 64, 32]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16]
    starts = [0, 0, 0]
    ends = [0, 0, 0]
    sorted_idx = [0, 0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=elem_counts,
        tensor_ggml_types=ggml_types,
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_sorted_tensor_index_random_valid_permutations() -> None:
    rng = random.Random(764069)

    for _ in range(300):
        count = rng.randint(1, 20)
        starts: list[int] = []
        ends: list[int] = []
        cursor = rng.randint(0, 32) * 32
        for _ in range(count):
            cursor += rng.randint(0, 4) * 32
            span = rng.randint(0, 6) * 32
            starts.append(cursor)
            ends.append(cursor + span)
            cursor += span

        perm = list(range(count))
        rng.shuffle(perm)

        err, bad = validate_sorted_tensor_index(starts, ends, perm)
        assert err == GGUF_TDBASE_OK
        assert bad == 0


def test_validate_sorted_tensor_index_random_adversarial_mutations() -> None:
    rng = random.Random(764077)

    for _ in range(240):
        count = rng.randint(3, 18)
        starts: list[int] = []
        ends: list[int] = []
        cursor = rng.randint(0, 32) * 32
        for _ in range(count):
            cursor += rng.randint(0, 3) * 32
            span = max(32, rng.randint(1, 6) * 32)
            starts.append(cursor)
            ends.append(cursor + span)
            cursor += span

        sorted_idx = list(range(count))
        case = rng.randint(0, 3)

        if case == 0:
            i = rng.randint(1, count - 1)
            starts[i], starts[i - 1] = starts[i - 1], starts[i]
            ends[i], ends[i - 1] = ends[i - 1], ends[i]
            err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
            assert err == GGUF_TDBASE_ERR_OVERLAP
            assert bad == i
        elif case == 1:
            i = rng.randint(1, count - 1)
            starts[i] = max(starts[i - 1], starts[i] - 32)
            ends[i - 1] = starts[i] + 32
            err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
            assert err == GGUF_TDBASE_ERR_OVERLAP
            assert bad == i
        elif case == 2:
            i = rng.randint(0, count - 1)
            sorted_idx[i] = count + rng.randint(0, 10)
            err, bad = validate_sorted_tensor_index(starts, ends, sorted_idx)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == i


def test_build_tensor_sorted_position_map_happy_path() -> None:
    sorted_idx = [2, 0, 3, 1]
    inverse_map = [0, 0, 0, 0]

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0
    assert inverse_map == [1, 3, 0, 2]


def test_build_tensor_sorted_position_map_empty_ok() -> None:
    err, bad = build_tensor_sorted_position_map([], [])
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_build_tensor_sorted_position_map_rejects_bad_output_len() -> None:
    err, bad = build_tensor_sorted_position_map([0, 1, 2], [0, 0])
    assert err == GGUF_TDBASE_ERR_NULL_PTR
    assert bad == 0


def test_build_tensor_sorted_position_map_rejects_out_of_range_index() -> None:
    sorted_idx = [0, 4, 1, 2]
    inverse_map = [0, 0, 0, 0]

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 1


def test_build_tensor_sorted_position_map_rejects_duplicate_index() -> None:
    sorted_idx = [0, 2, 1, 2]
    inverse_map = [0, 0, 0, 0]

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 3


def test_build_tensor_sorted_position_map_with_built_index() -> None:
    rel_offsets = [0xC0, 0x00, 0x40]
    elem_counts = [32, 64, 32]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16]
    starts = [0, 0, 0]
    ends = [0, 0, 0]
    sorted_idx = [0, 0, 0]
    inverse_map = [0, 0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=elem_counts,
        tensor_ggml_types=ggml_types,
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    for original_idx, sorted_pos in enumerate(inverse_map):
        assert sorted_idx[sorted_pos] == original_idx


def test_build_tensor_sorted_position_map_random_permutations() -> None:
    rng = random.Random(764091)

    for _ in range(400):
        count = rng.randint(1, 32)
        sorted_idx = list(range(count))
        rng.shuffle(sorted_idx)
        inverse_map = [0] * count

        err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        for original_idx, sorted_pos in enumerate(inverse_map):
            assert sorted_pos < count
            assert sorted_idx[sorted_pos] == original_idx


def test_build_tensor_sorted_position_map_random_adversarial() -> None:
    rng = random.Random(764101)

    for _ in range(320):
        count = rng.randint(2, 28)
        sorted_idx = list(range(count))
        rng.shuffle(sorted_idx)
        inverse_map = [0] * count

        if rng.randint(0, 1) == 0:
            i = rng.randint(0, count - 1)
            sorted_idx[i] = count + rng.randint(0, 7)
            err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == i
        else:
            i = rng.randint(1, count - 1)
            sorted_idx[i] = sorted_idx[rng.randint(0, i - 1)]
            err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == i


def test_validate_sorted_position_map_happy_path() -> None:
    sorted_idx = [2, 0, 3, 1]
    inverse_map = [1, 3, 0, 2]

    err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_sorted_position_map_empty_ok() -> None:
    err, bad = validate_sorted_position_map([], [])
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_sorted_position_map_rejects_bad_input_len() -> None:
    err, bad = validate_sorted_position_map([0, 1, 2], [0, 1])
    assert err == GGUF_TDBASE_ERR_NULL_PTR
    assert bad == 0


def test_validate_sorted_position_map_rejects_out_of_range_sorted_pos() -> None:
    sorted_idx = [0, 2, 1, 3]
    inverse_map = [0, 4, 1, 3]

    err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 1


def test_validate_sorted_position_map_rejects_duplicate_sorted_pos() -> None:
    sorted_idx = [2, 0, 3, 1]
    inverse_map = [1, 3, 1, 2]

    err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 2


def test_validate_sorted_position_map_rejects_roundtrip_mismatch() -> None:
    sorted_idx = [2, 0, 3, 1]
    inverse_map = [1, 2, 0, 3]

    err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 1


def test_validate_sorted_position_map_rejects_bad_sorted_tensor_index() -> None:
    sorted_idx = [2, 0, 5, 1]
    inverse_map = [1, 3, 0, 2]

    err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert bad == 3


def test_validate_sorted_position_map_with_built_index() -> None:
    rel_offsets = [0xC0, 0x00, 0x40]
    elem_counts = [32, 64, 32]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16]
    starts = [0, 0, 0]
    ends = [0, 0, 0]
    sorted_idx = [0, 0, 0]
    inverse_map = [0, 0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x2000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=elem_counts,
        tensor_ggml_types=ggml_types,
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0


def test_validate_sorted_position_map_random_permutations() -> None:
    rng = random.Random(764201)

    for _ in range(400):
        count = rng.randint(1, 32)
        sorted_idx = list(range(count))
        rng.shuffle(sorted_idx)
        inverse_map = [0] * count

        err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0


def test_validate_sorted_position_map_random_adversarial() -> None:
    rng = random.Random(764211)

    for _ in range(320):
        count = rng.randint(2, 28)
        sorted_idx = list(range(count))
        rng.shuffle(sorted_idx)
        inverse_map = [0] * count

        err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        case = rng.randint(0, 3)
        if case == 0:
            i = rng.randint(0, count - 1)
            inverse_map[i] = count + rng.randint(0, 7)
            err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == i
        elif case == 1:
            i = rng.randint(1, count - 1)
            inverse_map[i] = inverse_map[rng.randint(0, i - 1)]
            err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == i
        elif case == 2:
            i = rng.randint(0, count - 1)
            inverse_map[i] = (inverse_map[i] + 1) % count
            err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == i
        else:
            i = rng.randint(0, count - 1)
            sorted_pos = inverse_map[i]
            sorted_idx[sorted_pos] = count + rng.randint(0, 7)
            err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert 0 <= bad < count


def test_tensor_index_to_abs_range_happy_path() -> None:
    starts = [0x1200, 0x1400, 0x1500, 0x1700]
    ends = [0x1280, 0x1480, 0x1680, 0x17C0]
    sorted_idx = [2, 0, 3, 1]
    inverse_map = [0, 0, 0, 0]

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, abs_start, abs_end = tensor_index_to_abs_range(0, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert (abs_start, abs_end) == (0x1400, 0x1480)

    err, abs_start, abs_end = tensor_index_to_abs_range(1, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert (abs_start, abs_end) == (0x1700, 0x17C0)

    err, abs_start, abs_end = tensor_index_to_abs_range(2, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert (abs_start, abs_end) == (0x1200, 0x1280)

    err, abs_start, abs_end = tensor_index_to_abs_range(3, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert (abs_start, abs_end) == (0x1500, 0x1680)


def test_tensor_index_to_abs_range_rejects_bad_lengths() -> None:
    err, abs_start, abs_end = tensor_index_to_abs_range(
        tensor_index=0,
        tensor_abs_starts=[0x1000, 0x1200],
        tensor_abs_ends=[0x1100],
        sorted_position_by_tensor=[0, 1],
    )
    assert err == GGUF_TDBASE_ERR_NULL_PTR
    assert abs_start == 0
    assert abs_end == 0

    err, abs_start, abs_end = tensor_index_to_abs_range(
        tensor_index=0,
        tensor_abs_starts=[0x1000, 0x1200],
        tensor_abs_ends=[0x1100, 0x1300],
        sorted_position_by_tensor=[0],
    )
    assert err == GGUF_TDBASE_ERR_NULL_PTR
    assert abs_start == 0
    assert abs_end == 0


def test_tensor_index_to_abs_range_rejects_out_of_bounds_index() -> None:
    starts = [0x1000, 0x1100, 0x1300]
    ends = [0x1080, 0x1180, 0x1380]
    sorted_idx = [1, 2, 0]
    inverse_map = [0, 0, 0]

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, abs_start, abs_end = tensor_index_to_abs_range(3, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert abs_start == 0
    assert abs_end == 0


def test_tensor_index_to_abs_range_rejects_bad_inverse_map_slot() -> None:
    starts = [0x1000, 0x1100, 0x1300]
    ends = [0x1080, 0x1180, 0x1380]
    inverse_map = [2, 4, 0]

    err, abs_start, abs_end = tensor_index_to_abs_range(1, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
    assert abs_start == 0
    assert abs_end == 0


def test_tensor_index_to_abs_range_rejects_malformed_range() -> None:
    starts = [0x1000, 0x1300, 0x1400]
    ends = [0x1080, 0x1200, 0x1480]
    inverse_map = [0, 1, 2]

    err, abs_start, abs_end = tensor_index_to_abs_range(1, starts, ends, inverse_map)
    assert err == GGUF_TDBASE_ERR_OVERFLOW
    assert abs_start == 0
    assert abs_end == 0


def test_tensor_index_to_abs_range_with_built_offset_index() -> None:
    rel_offsets = [0xC0, 0x00, 0x40, 0x100]
    elem_counts = [32, 64, 32, 128]
    ggml_types = [GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16, GGML_TYPE_F32]
    starts = [0, 0, 0, 0]
    ends = [0, 0, 0, 0]
    sorted_idx = [0, 0, 0, 0]
    inverse_map = [0, 0, 0, 0]

    err, bad = tensor_info_build_offset_index(
        tensor_data_base=0x1000,
        alignment=32,
        gguf_file_nbytes=0x4000,
        tensor_rel_offsets=rel_offsets,
        tensor_element_counts=elem_counts,
        tensor_ggml_types=ggml_types,
        out_abs_starts=starts,
        out_abs_ends=ends,
        out_sorted_tensor_indices=sorted_idx,
    )
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
    assert err == GGUF_TDBASE_OK
    assert bad == 0

    for original_idx in range(len(rel_offsets)):
        err, abs_start, abs_end = tensor_index_to_abs_range(
            original_idx,
            starts,
            ends,
            inverse_map,
        )
        assert err == GGUF_TDBASE_OK
        sorted_pos = inverse_map[original_idx]
        assert abs_start == starts[sorted_pos]
        assert abs_end == ends[sorted_pos]


def test_inverse_map_and_range_index_random_permutation_parity() -> None:
    rng = random.Random(764311)

    for _ in range(220):
        count = rng.randint(1, 36)
        alignment = 1 << rng.randint(5, 7)
        base = rng.randint(0x2000, 0xA000)
        base = ((base + alignment - 1) // alignment) * alignment

        sorted_rel_offsets: list[int] = []
        sorted_elem_counts: list[int] = []
        sorted_types: list[int] = []
        sorted_nbytes: list[int] = []

        rel_cursor = 0
        for _idx in range(count):
            rel_cursor += rng.randint(0, 5) * alignment

            ggml_type = rng.choice([GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0])
            if ggml_type == GGML_TYPE_Q4_0 or ggml_type == GGML_TYPE_Q8_0:
                elem_count = 32 * rng.randint(1, 8)
            else:
                elem_count = rng.choice([16, 32, 48, 64, 80, 96, 128, 160])

            err, nbytes = tensor_bytes_for_type(ggml_type, elem_count)
            assert err == GGUF_TDBASE_OK

            sorted_rel_offsets.append(rel_cursor)
            sorted_elem_counts.append(elem_count)
            sorted_types.append(ggml_type)
            sorted_nbytes.append(nbytes)

            rel_cursor += ((nbytes + alignment - 1) // alignment) * alignment

        perm = list(range(count))
        rng.shuffle(perm)

        rel_offsets = [sorted_rel_offsets[p] for p in perm]
        elem_counts = [sorted_elem_counts[p] for p in perm]
        ggml_types = [sorted_types[p] for p in perm]
        expected_nbytes = [sorted_nbytes[p] for p in perm]

        starts = [0] * count
        ends = [0] * count
        sorted_idx = [0] * count

        err, bad = tensor_info_build_offset_index(
            tensor_data_base=base,
            alignment=alignment,
            gguf_file_nbytes=base + rel_cursor + 0x200,
            tensor_rel_offsets=rel_offsets,
            tensor_element_counts=elem_counts,
            tensor_ggml_types=ggml_types,
            out_abs_starts=starts,
            out_abs_ends=ends,
            out_sorted_tensor_indices=sorted_idx,
        )
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        inverse_map = [0] * count
        err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        err, bad = validate_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        for original_idx in range(count):
            err, abs_start, abs_end = tensor_index_to_abs_range(
                tensor_index=original_idx,
                tensor_abs_starts=starts,
                tensor_abs_ends=ends,
                sorted_position_by_tensor=inverse_map,
            )
            assert err == GGUF_TDBASE_OK
            assert abs_start == base + rel_offsets[original_idx]
            assert abs_end == abs_start + expected_nbytes[original_idx]

            err, found_idx = tensor_range_find_index_by_abs_offset(
                abs_start,
                starts,
                ends,
                sorted_idx,
            )
            assert err == GGUF_TDBASE_OK
            assert found_idx == original_idx

            if abs_end > abs_start:
                err, found_idx = tensor_range_find_index_by_abs_offset(
                    abs_end - 1,
                    starts,
                    ends,
                    sorted_idx,
                )
                assert err == GGUF_TDBASE_OK
                assert found_idx == original_idx


def test_inverse_map_and_range_index_random_adversarial_cases() -> None:
    rng = random.Random(764313)

    for _ in range(160):
        count = rng.randint(3, 24)
        sorted_idx = list(range(count))
        rng.shuffle(sorted_idx)

        inverse_map = [0] * count
        err, bad = build_tensor_sorted_position_map(sorted_idx, inverse_map)
        assert err == GGUF_TDBASE_OK
        assert bad == 0

        starts = []
        ends = []
        cursor = rng.randint(0x1000, 0x3000)
        for _i in range(count):
            span = rng.randint(1, 128)
            starts.append(cursor)
            ends.append(cursor + span)
            cursor += span + rng.randint(0, 32)

        case = rng.randint(0, 3)
        if case == 0:
            victim = rng.randint(0, count - 1)
            inverse_bad = list(inverse_map)
            inverse_bad[victim] = count + rng.randint(0, 16)

            err, bad = validate_sorted_position_map(sorted_idx, inverse_bad)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == victim

            err, abs_start, abs_end = tensor_index_to_abs_range(
                victim,
                starts,
                ends,
                inverse_bad,
            )
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert abs_start == 0
            assert abs_end == 0
        elif case == 1:
            victim = rng.randint(1, count - 1)
            inverse_bad = list(inverse_map)
            inverse_bad[victim] = inverse_bad[rng.randint(0, victim - 1)]

            err, bad = validate_sorted_position_map(sorted_idx, inverse_bad)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert bad == victim
        elif case == 2:
            sorted_pos = rng.randint(0, count - 1)
            original_idx = sorted_idx[sorted_pos]
            starts_bad = list(starts)
            ends_bad = list(ends)
            ends_bad[sorted_pos] = starts_bad[sorted_pos] - 1

            err, abs_start, abs_end = tensor_index_to_abs_range(
                original_idx,
                starts_bad,
                ends_bad,
                inverse_map,
            )
            assert err == GGUF_TDBASE_ERR_OVERFLOW
            assert abs_start == 0
            assert abs_end == 0
        else:
            sorted_bad = list(sorted_idx)
            i = rng.randint(0, count - 1)
            sorted_bad[i] = count + rng.randint(0, 12)

            err, bad = validate_sorted_position_map(sorted_bad, inverse_map)
            assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS
            assert 0 <= bad < count






def run() -> None:
    test_default_alignment_examples()
    test_tensor_bytes_for_type_scalars()
    test_tensor_bytes_for_type_quantized_exact_blocks()
    test_tensor_bytes_for_type_reject_non_multiple_block()
    test_tensor_bytes_for_type_reject_unknown_type()
    test_tensor_bytes_for_type_overflow_guard()
    test_tensor_type_block_size_known_types()
    test_tensor_type_block_bytes_known_types()
    test_tensor_type_helpers_reject_unknown_type()
    test_custom_power_of_two_alignment()
    test_reject_non_power_of_two_alignment()
    test_overflow_guard_near_u64_max()
    test_random_alignment_matches_reference_formula()
    test_tensor_data_base_offset_happy_path()
    test_tensor_data_base_offset_default_alignment()
    test_tensor_data_base_offset_reject_bad_alignment()
    test_tensor_data_base_offset_reject_misaligned_base()
    test_tensor_data_base_offset_reject_misaligned_rel()
    test_tensor_data_base_offset_overflow_guard()
    test_tensor_data_base_offset_random_aligned_cases()
    test_tensor_resolve_range_happy_path()
    test_tensor_resolve_range_default_alignment()
    test_tensor_resolve_range_reject_bad_alignment()
    test_tensor_resolve_range_reject_misaligned_base()
    test_tensor_resolve_range_reject_misaligned_offset()
    test_tensor_resolve_range_overflow_guard()
    test_tensor_resolve_range_out_of_bounds()
    test_tensor_resolve_range_allows_exact_eof()
    test_tensor_resolve_range_random_valid_cases()
    test_validate_tensor_ranges_non_overlapping_happy_path()
    test_validate_tensor_ranges_detect_overlap()
    test_validate_tensor_ranges_propagates_bad_alignment()
    test_validate_tensor_ranges_propagates_out_of_bounds_with_index()
    test_validate_tensor_ranges_random_non_overlapping()
    test_validate_tensor_ranges_random_overlap_cases()
    test_validate_tensor_ranges_sorted_non_overlapping_happy_path()
    test_validate_tensor_ranges_sorted_tie_breaks_equal_offsets_by_size()
    test_validate_tensor_ranges_sorted_detect_overlap()
    test_validate_tensor_ranges_sorted_propagates_bad_alignment_without_sort()
    test_validate_tensor_ranges_sorted_random_matches_quadratic_validator()
    test_tensor_info_resolve_byte_spans_happy_path()
    test_tensor_info_resolve_byte_spans_reject_bad_block_multiple()
    test_tensor_info_resolve_byte_spans_reject_unknown_type()
    test_tensor_info_resolve_byte_spans_detect_overlap_after_sizing()
    test_tensor_info_resolve_abs_ranges_happy_path()
    test_tensor_info_resolve_abs_ranges_propagates_bad_block_multiple()
    test_tensor_info_resolve_abs_ranges_reject_null_outputs()
    test_tensor_range_find_by_abs_offset_happy_path()
    test_tensor_range_find_by_abs_offset_gap_and_end_exclusive()
    test_tensor_range_find_by_abs_offset_rejects_malformed_range()
    test_tensor_range_find_by_abs_offset_empty_table()
    test_tensor_range_find_by_rel_offset_happy_path()
    test_tensor_range_find_by_rel_offset_propagates_alignment_error()
    test_tensor_range_find_by_rel_offset_propagates_misaligned_rel()
    test_tensor_range_find_by_rel_offset_gap_and_out_of_bounds()
    test_tensor_range_find_by_rel_offset_default_alignment_happy_path()
    test_tensor_range_find_by_rel_offset_default_alignment_propagates_error()
    test_tensor_range_find_index_by_abs_offset_happy_path()
    test_tensor_range_find_index_by_abs_offset_rejects_bad_index_table()
    test_tensor_range_find_index_by_abs_offset_gap_propagates_miss()
    test_tensor_range_find_index_by_rel_offset_happy_path()
    test_tensor_range_find_index_by_rel_offset_propagates_alignment_error()
    test_tensor_range_find_index_by_rel_offset_with_built_index()
    test_tensor_range_find_index_by_rel_offset_default_happy_path()
    test_tensor_range_find_index_by_rel_offset_default_propagates_error()
    test_tensor_info_build_offset_index_happy_path()
    test_tensor_info_build_offset_index_propagates_bad_block_multiple()
    test_tensor_info_build_offset_index_detects_overlap()
    test_tensor_info_build_offset_index_random_non_overlapping()
    test_validate_sorted_tensor_index_happy_path()
    test_validate_sorted_tensor_index_empty_ok()
    test_validate_sorted_tensor_index_detects_overflow_range()
    test_validate_sorted_tensor_index_detects_unsorted_starts()
    test_validate_sorted_tensor_index_detects_overlap()
    test_validate_sorted_tensor_index_rejects_bad_mapped_index()
    test_validate_sorted_tensor_index_rejects_duplicate_mapped_index()
    test_validate_sorted_tensor_index_with_built_index()
    test_validate_sorted_tensor_index_random_valid_permutations()
    test_validate_sorted_tensor_index_random_adversarial_mutations()
    test_build_tensor_sorted_position_map_happy_path()
    test_build_tensor_sorted_position_map_empty_ok()
    test_build_tensor_sorted_position_map_rejects_bad_output_len()
    test_build_tensor_sorted_position_map_rejects_out_of_range_index()
    test_build_tensor_sorted_position_map_rejects_duplicate_index()
    test_build_tensor_sorted_position_map_with_built_index()
    test_build_tensor_sorted_position_map_random_permutations()
    test_build_tensor_sorted_position_map_random_adversarial()
    test_validate_sorted_position_map_happy_path()
    test_validate_sorted_position_map_empty_ok()
    test_validate_sorted_position_map_rejects_bad_input_len()
    test_validate_sorted_position_map_rejects_out_of_range_sorted_pos()
    test_validate_sorted_position_map_rejects_duplicate_sorted_pos()
    test_validate_sorted_position_map_rejects_roundtrip_mismatch()
    test_validate_sorted_position_map_rejects_bad_sorted_tensor_index()
    test_validate_sorted_position_map_with_built_index()
    test_validate_sorted_position_map_random_permutations()
    test_validate_sorted_position_map_random_adversarial()
    test_tensor_index_to_abs_range_happy_path()
    test_tensor_index_to_abs_range_rejects_bad_lengths()
    test_tensor_index_to_abs_range_rejects_out_of_bounds_index()
    test_tensor_index_to_abs_range_rejects_bad_inverse_map_slot()
    test_tensor_index_to_abs_range_rejects_malformed_range()
    test_tensor_index_to_abs_range_with_built_offset_index()
    test_inverse_map_and_range_index_random_permutation_parity()
    test_inverse_map_and_range_index_random_adversarial_cases()
    print("gguf_tensor_data_base_reference_checks=ok")


if __name__ == "__main__":
    run()
