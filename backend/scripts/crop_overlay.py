#!/usr/bin/env python3
"""
Diagnostic: render the source video with supoclip's chosen crop window
drawn on top, so you can see WHAT the per-shot reframe algorithm targeted
in the original frame.

Born of ENG-5719 — the vertical reframe was cropping faces out of frame,
and inspection of the existing CloudWatch logs (which only emit x= per shot)
wasn't enough to tell whether the X target was landing on a real face or
on a TV-screen logo / lighting fixture / piece of furniture. This script
consumes the structured `*.reframe_plan.json` sidecar that
`render_reframed_clip_ffmpeg` now writes next to every per-shot-cut clip
and renders an annotated MP4 you can scrub through.

Usage
-----

# Most common path: you already have the sidecar JSON locally
python crop_overlay.py \
    --plan ./ed5270c6.reframe_plan.json \
    --source ./original.mp4 \
    --output ./ed5270c6-overlay.mp4

# Fetch sidecar from a running supoclip backend (no auth header support yet —
# best for local dev / port-forwarded prod where the trusted-frontend header
# isn't required)
python crop_overlay.py \
    --task-id 364f8a6c-021c-47ec-8408-e5ad3cd3ff79 \
    --clip-id ed5270c6-1661-49f7-b06e-67b1cf4754bc \
    --base-url http://localhost:8000 \
    --source ./original.mp4 \
    --output ./ed5270c6-overlay.mp4

What you'll see in the output
-----------------------------

- A yellow rectangle (the supoclip 1080×1920-equivalent crop window scaled
  to the source's coordinate system) that hard-cuts position at every
  per-shot segment boundary.
- Red dots at every face center the DNN detector returned (with confidence
  labels — these are the "26 faces detected" you see in the logs).
- Cyan vertical lines at scene-cut times.
- White text in the top-left summarising the active shot (index, mode,
  face_count, x_target).

If the yellow rectangle is NOT centred on a real face during a shot, the
clustering/selection step is the culprit. If the red dots include obvious
non-face features (TV logos, furniture corners), the DNN detector itself
is the culprit.

Dependencies
------------

- ffmpeg in PATH (used for the actual rendering).
- httpx (optional — only needed when fetching the sidecar from a backend).
  `uv pip install httpx` if you need it.

Notes
-----

- This is a diagnostic tool, not part of the rendered-output pipeline.
  It writes to wherever you point --output and does not touch the
  supoclip cluster or any S3 bucket.
- The `pan`, `vertical_split`, and static fallback reframe paths do NOT
  write a debug sidecar (only the per-shot `cut` path does). If the
  endpoint returns 404 with "non-cut reframe mode" in the detail, your
  clip didn't go through the path we're trying to debug.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("crop_overlay")


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


def load_plan_from_file(path: Path) -> Dict[str, Any]:
    """Load the *.reframe_plan.json sidecar from a local file."""
    with open(path, "r") as f:
        data = json.load(f)
    schema = data.get("schema_version")
    if schema != 1:
        log.warning(
            "Plan schema_version=%s differs from this script's expected v1 — "
            "shape may have drifted, output may be wrong",
            schema,
        )
    return data


def fetch_plan_from_backend(base_url: str, task_id: str, clip_id: str) -> Dict[str, Any]:
    """Pull the sidecar from a running supoclip backend.

    Auth notes: this hits the /reframe-plan endpoint, which uses
    `_require_task_owner` and reads trusted-frontend headers. For a local
    dev backend (docker-compose) these headers are usually unenforced; for
    a port-forwarded prod backend you'd need to also pass the appropriate
    user-id header. We deliberately keep this thin — pull, parse, return.
    Anything fancier and you may as well curl + --plan-file.
    """
    try:
        import httpx
    except ImportError:
        raise SystemExit(
            "httpx is required for --base-url mode. Install with `pip install httpx`, "
            "or download the sidecar with `curl ... > plan.json` and pass --plan-file."
        )

    url = f"{base_url.rstrip('/')}/tasks/{task_id}/clips/{clip_id}/reframe-plan"
    log.info("Fetching plan from %s", url)
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        if resp.status_code == 404:
            raise SystemExit(
                f"Backend returned 404. Either the clip used a non-cut "
                f"reframe mode (vertical_pan / vertical_split / static fallback "
                f"all skip the sidecar), or the sidecar was wiped on container "
                f"restart (supoclip /tmp is per-task). Detail: {resp.text}"
            )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# ffmpeg filtergraph builder
# ---------------------------------------------------------------------------


# Aesthetic constants — tuned so the overlay stays readable on dark / busy
# video without obscuring the underlying frame.
_CROP_BOX_COLOR = "yellow"
_CROP_BOX_THICKNESS = 4
_SCENE_CUT_COLOR = "cyan"
_SCENE_CUT_THICKNESS = 2
_FACE_DOT_COLOR = "red"
_FACE_DOT_RADIUS = 8
_LABEL_FG = "white"
_LABEL_BG = "black@0.55"  # semi-transparent so video shows through
_LABEL_FONTSIZE = 22


def _ff_time(value: float) -> str:
    """Format seconds for ffmpeg `enable` expressions; trim trailing zeros so
    the filtergraph stays readable when an editor opens the rendered output's
    metadata."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _ff_str(value: str) -> str:
    """Escape a string for embedding inside an ffmpeg filter option value.

    Filter option values are colon/comma-delimited so we have to escape `:`
    and `\\` ourselves; single-quote wrapping handles the rest. drawtext text
    is additionally subject to its own escape rules — we keep things simple
    by only using ASCII labels and avoiding `:` / `\\` in the text itself.
    """
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_filtergraph(plan: Dict[str, Any]) -> str:
    """Compose the ffmpeg `-vf` filtergraph for the overlay.

    Structure:
        1. One drawbox per merged_timeline entry → the crop window during
           that segment.
        2. One drawbox vertical line per scene_cut_time → editor cuts.
        3. One drawbox per face center per shot → where the DNN detector
           thought a face was.
        4. One drawtext per shot → the shot's metadata (mode, face_count,
           x_target) in the corner during that shot.

    All filters use `enable=between(t,start,end)` to scope themselves to
    the right time window. ffmpeg evaluates these per-frame so the overhead
    is bounded by output frame count, not segment count.
    """
    filters: List[str] = []

    crop_w = int(plan["crop_w"])
    crop_h = int(plan["crop_h"])

    # 1. Crop window per merged segment. This is what render_reframed_clip_ffmpeg
    # is doing — we draw the box and the user can compare it against face
    # positions in the source.
    for seg in plan.get("merged_timeline", []):
        start = float(seg["start"])
        end = float(seg["end"])
        x = int(seg["x"])
        filters.append(
            f"drawbox=x={x}:y=0:w={crop_w}:h={crop_h}"
            f":color={_CROP_BOX_COLOR}:thickness={_CROP_BOX_THICKNESS}"
            f":enable='between(t,{_ff_time(start)},{_ff_time(end)})'"
        )

    # 2. Scene-cut vertical lines (visible during a short window around the
    # cut — long enough to spot when scrubbing, short enough not to clutter).
    for cut_time in plan.get("scene_cut_times", []):
        start = max(0.0, float(cut_time) - 0.15)
        end = float(cut_time) + 0.45
        filters.append(
            f"drawbox=x=0:y=0:w={_SCENE_CUT_THICKNESS}:h=ih"
            f":color={_SCENE_CUT_COLOR}:thickness=fill"
            f":enable='between(t,{_ff_time(start)},{_ff_time(end)})'"
        )

    # 3. Face-detection dots per shot. Each shot's `face_centers` list is
    # what the DNN+Haar pipeline returned (after outlier filtering). The
    # diameter is `_FACE_DOT_RADIUS*2` and we render via drawbox (drawcircle
    # is not part of standard ffmpeg builds).
    for shot in plan.get("shots", []):
        s_start = float(shot["start"])
        s_end = float(shot["end"])
        enable = f"between(t,{_ff_time(s_start)},{_ff_time(s_end)})"
        for fc in shot.get("face_centers", []):
            cx = int(fc["center_x"])
            cy = int(fc["center_y"])
            x0 = max(0, cx - _FACE_DOT_RADIUS)
            y0 = max(0, cy - _FACE_DOT_RADIUS)
            filters.append(
                f"drawbox=x={x0}:y={y0}:w={_FACE_DOT_RADIUS * 2}:h={_FACE_DOT_RADIUS * 2}"
                f":color={_FACE_DOT_COLOR}:thickness=fill"
                f":enable='{enable}'"
            )

    # 4. Per-shot label in the top-left. Includes shot index, mode, face
    # count, x_target — the same fields you'd want when correlating the
    # overlay against the CloudWatch log line for the shot.
    for shot in plan.get("shots", []):
        s_start = float(shot["start"])
        s_end = float(shot["end"])
        mode = shot.get("mode", "?")
        face_count = shot.get("face_count", 0)
        x_target = shot.get("x_target")
        x_label = "n/a" if x_target is None else str(x_target)
        label = (
            f"shot {shot.get('shot_index', '?')} "
            f"{mode} faces={face_count} x={x_label}"
        )
        filters.append(
            f"drawtext=text='{_ff_str(label)}'"
            f":x=20:y=20:fontsize={_LABEL_FONTSIZE}:fontcolor={_LABEL_FG}"
            f":box=1:boxcolor={_LABEL_BG}:boxborderw=8"
            f":enable='between(t,{_ff_time(s_start)},{_ff_time(s_end)})'"
        )

    return ",".join(filters)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def render_overlay(source_path: Path, plan: Dict[str, Any], output_path: Path) -> None:
    """Invoke ffmpeg with the composed filtergraph."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH. Install ffmpeg first.")

    filtergraph = build_filtergraph(plan)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        filtergraph,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",  # don't re-encode audio; we only annotated video
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    log.info("Running: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # ffmpeg's stderr is the useful part — surface it verbatim so the
        # user can see exactly which filter parse failed if there's a bug
        # in build_filtergraph.
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"ffmpeg failed with exit code {proc.returncode}")
    log.info("Wrote overlay to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="Path to a previously downloaded *.reframe_plan.json sidecar.",
    )
    parser.add_argument(
        "--task-id",
        help="Supoclip task ID (use with --clip-id and --base-url to fetch).",
    )
    parser.add_argument(
        "--clip-id",
        help="Supoclip clip ID.",
    )
    parser.add_argument(
        "--base-url",
        help="Supoclip backend base URL, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source video the clip was cut from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./crop_overlay.mp4"),
        help="Path to write the annotated MP4 (default: ./crop_overlay.mp4).",
    )
    args = parser.parse_args()
    if not args.plan and not (args.task_id and args.clip_id and args.base_url):
        parser.error(
            "Provide either --plan <file> OR all of (--task-id, --clip-id, --base-url)."
        )
    return args


def main() -> None:
    args = parse_args()
    if args.plan:
        plan = load_plan_from_file(args.plan)
    else:
        plan = fetch_plan_from_backend(args.base_url, args.task_id, args.clip_id)

    if plan.get("mode") != "cut":
        # The sidecar exists only for cut mode (per-shot hard cuts), but we
        # double-check here in case the file came from a future schema or
        # was hand-edited. If we get here the overlay would render an empty
        # filtergraph; bail with a useful message instead.
        log.warning(
            "Plan mode=%r is not 'cut' — overlay may be empty / inaccurate. "
            "Only the per-shot cut path collects debug data today.",
            plan.get("mode"),
        )

    if not args.source.exists():
        raise SystemExit(f"Source video not found: {args.source}")

    render_overlay(args.source, plan, args.output)


if __name__ == "__main__":
    main()
