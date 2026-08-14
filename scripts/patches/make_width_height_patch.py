#!/usr/bin/env python3
"""Edit the sglang source in place to add target.width/height, then `git diff` gives the patch.

Run inside the container:
    docker exec h3 python3 /patches/make_width_height_patch.py            # apply
    docker exec h3 python3 /patches/make_width_height_patch.py --reverse  # un-apply

Why a generator instead of a hand-written diff: the two edits are large and indentation-sensitive,
and a diff with a stale line number silently fails to apply. Generating from anchors that must
match exactly (each replacement asserts it found its anchor once) means the patch either builds
from the real source or the script refuses.

Why --reverse exists: this patch has to be diffed *on top of* the short-edge patch, because both
edit `_validate_target` in request_validation.py. A plain `git diff` against the image's HEAD would
fold the short-edge change into this patch, and then applying both in sequence conflicts. So
`make_patch.sh` reverses this edit, commits the short-edge state as a temporary baseline, re-applies
forward, and diffs against that.
"""
import sys

ROOT = "/sgl-workspace/sglang/python/sglang/multimodal_gen"
VALIDATION = f"{ROOT}/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py"
SAMPLE = f"{ROOT}/configs/sample/minimax_h3.py"

IMPORT_OLD = """from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_allowed_short_edges,
)"""
IMPORT_NEW = """from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_MAX_PIXELS,
    minimax_h3_allowed_short_edges,
)"""

TARGET_OLD = '''    short_edge = _require_int(target.get("short_edge"), f"{path}.short_edge")
    allowed = minimax_h3_allowed_short_edges()
    if short_edge not in allowed:
        # Identical to the previous message when nothing is opted in, so the released
        # deployment's error text does not change.
        expected = allowed[0] if len(allowed) == 1 else f"one of {list(allowed)}"
        raise ValueError(
            f"{path}.short_edge must be {expected} for minimax_h3, got {short_edge}"
        )
    aspect_ratio = _require_str(target.get("aspect_ratio"), f"{path}.aspect_ratio")
    if profile.aspect_ratio_forced_auto and aspect_ratio != "auto":
        raise ValueError(
            f'{path}.aspect_ratio must be "auto" for task {profile.task!r}, '
            f"got {aspect_ratio!r}"
        )
    has_duration = target.get("duration_seconds") is not None
    if (
        profile.task in {MINIMAX_H3_TASK_T2VA, MINIMAX_H3_TASK_REF2VA}
        and aspect_ratio != "auto"
        and aspect_ratio not in MINIMAX_H3_FINITE_ASPECT_RATIOS
    ):
        raise ValueError(
            f"{path}.aspect_ratio for task {profile.task!r} must be 'auto' or "
            f"one of {list(MINIMAX_H3_FINITE_ASPECT_RATIOS)!r}, got "
            f"{aspect_ratio!r}"
        )
'''

TARGET_NEW = '''    # Geometry arrives as one of two mutually exclusive groups: width+height names an exact
    # canvas, while the released short_edge+aspect_ratio pair names a policy ratio and keeps its
    # original behaviour byte-for-byte, including its error messages.
    explicit_canvas = target.get("width") is not None or target.get("height") is not None
    if explicit_canvas:
        if target.get("short_edge") is not None or target.get("aspect_ratio") is not None:
            raise ValueError(
                f"{path} accepts either width+height or short_edge+aspect_ratio, not both"
            )
        if profile.aspect_ratio_forced_auto:
            raise ValueError(
                f"{path}.width/height are not allowed for task {profile.task!r}, which "
                'requires aspect_ratio "auto"'
            )
        width = _require_int(target.get("width"), f"{path}.width")
        height = _require_int(target.get("height"), f"{path}.height")
        for field_name, value in (("width", width), ("height", height)):
            if value <= 0 or value % MINIMAX_H3_CANVAS_MULTIPLE:
                # Refuse rather than round. This is the same rule the short-edge allowlist
                # states: a request that silently changes geometry is worse than one refused.
                raise ValueError(
                    f"{path}.{field_name} must be a positive multiple of "
                    f"{MINIMAX_H3_CANVAS_MULTIPLE}, got {value}"
                )
        if width * height > MINIMAX_H3_MAX_PIXELS:
            # The resolver's soft area cap would scale an oversized canvas down. That is right
            # for a ratio request but wrong for an exact one, so refuse instead.
            raise ValueError(
                f"{path}.width*height must be at most {MINIMAX_H3_MAX_PIXELS} px, got "
                f"{width * height} for {width}x{height}"
            )
        short_edge = min(width, height)
        allowed = minimax_h3_allowed_short_edges()
        if short_edge not in allowed:
            expected = allowed[0] if len(allowed) == 1 else f"one of {list(allowed)}"
            raise ValueError(
                f"{path}: min(width, height) is the short edge and must be {expected} for "
                f"minimax_h3, got {short_edge} from {width}x{height}"
            )
        # An exact canvas bypasses the finite-ratio allowlist on purpose: that list constrains
        # *ratio* requests to tested shapes, and a caller naming both axes has already chosen.
        # The resolver still enforces the inclusive 1:4..4:1 range on the derived ratio.
        aspect_ratio = f"{width}:{height}"
        has_duration = target.get("duration_seconds") is not None
    else:
        short_edge = _require_int(target.get("short_edge"), f"{path}.short_edge")
        allowed = minimax_h3_allowed_short_edges()
        if short_edge not in allowed:
            # Identical to the previous message when nothing is opted in, so the released
            # deployment's error text does not change.
            expected = allowed[0] if len(allowed) == 1 else f"one of {list(allowed)}"
            raise ValueError(
                f"{path}.short_edge must be {expected} for minimax_h3, got {short_edge}"
            )
        aspect_ratio = _require_str(target.get("aspect_ratio"), f"{path}.aspect_ratio")
        if profile.aspect_ratio_forced_auto and aspect_ratio != "auto":
            raise ValueError(
                f'{path}.aspect_ratio must be "auto" for task {profile.task!r}, '
                f"got {aspect_ratio!r}"
            )
        has_duration = target.get("duration_seconds") is not None
        if (
            profile.task in {MINIMAX_H3_TASK_T2VA, MINIMAX_H3_TASK_REF2VA}
            and aspect_ratio != "auto"
            and aspect_ratio not in MINIMAX_H3_FINITE_ASPECT_RATIOS
        ):
            raise ValueError(
                f"{path}.aspect_ratio for task {profile.task!r} must be 'auto' or "
                f"one of {list(MINIMAX_H3_FINITE_ASPECT_RATIOS)!r}, got "
                f"{aspect_ratio!r}"
            )
'''

PROJECTION_OLD = '''                for field_name in (
                    "short_edge",
                    "aspect_ratio",
                    "duration_seconds",
                )'''

PROJECTION_NEW = '''                for field_name in (
                    "short_edge",
                    "aspect_ratio",
                    # width/height are the exact-canvas alternative to short_edge+aspect_ratio.
                    # Without them here the projection silently drops both, and the request is
                    # then rejected for a missing short_edge -- confusing but not dangerous.
                    "width",
                    "height",
                    "duration_seconds",
                )'''


REVERSE = "--reverse" in sys.argv[1:]


def replace_once(path, old, new, label):
    if REVERSE:
        old, new = new, old
    with open(path) as handle:
        text = handle.read()
    if new in text and old not in text:
        print(f"ALREADY {label}")
        return
    count = text.count(old)
    if count != 1:
        sys.exit(f"REFUSING {label}: anchor found {count} times in {path}, expected exactly 1")
    with open(path, "w") as handle:
        handle.write(text.replace(old, new, 1))
    print(f"{'REVERTED' if REVERSE else 'EDITED  '} {label}")


replace_once(VALIDATION, IMPORT_OLD, IMPORT_NEW, "request_validation.py import")
replace_once(VALIDATION, TARGET_OLD, TARGET_NEW, "request_validation.py _validate_target")
replace_once(SAMPLE, PROJECTION_OLD, PROJECTION_NEW, "configs/sample/minimax_h3.py projection")
print("OK")
