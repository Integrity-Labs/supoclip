"""
Tests for reject_static_face_clusters (ENG-5807).

The function reads frames from a video file via cv2.VideoCapture. We can't
ship a video binary in the test suite, so we patch VideoCapture to return
canned numpy frames. The canned frames are designed to exhibit:

  * Static patches at known cluster bboxes (a painting on the wall) — pixel
    values are constant across all sampled frames → max pairwise mean-abs-
    diff ≈ 0, well below the 2.0 threshold → cluster should be dropped.
  * Moving patches at known cluster bboxes (a talking head) — pixel values
    cycle through distinct intensities across frames → max pairwise mean-
    abs-diff well above threshold → cluster should be kept.

Frame size 200×200 keeps the diffs cheap to compute. Cluster bboxes are
sized so the sqrt(area) reconstruction lands cleanly inside a known region.

Calibration data captured against BN job 2623b681 (Supoclip task
07140864) — see clip 1 shot 4 fixture below for the production-realistic
case.
"""

from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from src.video_utils import reject_static_face_clusters


# --- helpers ---------------------------------------------------------------


def _frame_with_patches(
    *,
    width: int = 200,
    height: int = 200,
    static_regions: List[tuple] = (),  # (x0, y0, x1, y1, value)
    moving_regions: List[tuple] = (),  # (x0, y0, x1, y1, value)
) -> np.ndarray:
    """Build a synthetic BGR frame with the given patches.

    Background is mid-grey (128). `static_regions` are filled with the given
    fixed value across all frames a test produces. `moving_regions` are filled
    with the value the test passes in for THIS frame; subsequent frames pass
    different values to simulate motion.
    """
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    for x0, y0, x1, y1, v in static_regions:
        frame[y0:y1, x0:x1] = v
    for x0, y0, x1, y1, v in moving_regions:
        frame[y0:y1, x0:x1] = v
    return frame


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture that returns canned frames.

    Frames are looked up by the most-recent CAP_PROP_POS_MSEC seek; we map
    each seek to the next frame in the supplied list. This matches how
    reject_static_face_clusters drives the capture — one seek per sample
    time, then read().
    """

    def __init__(self, frames: List[np.ndarray]):
        self._frames = frames
        self._idx = 0

    def isOpened(self) -> bool:
        return True

    def set(self, prop, value):  # noqa: D401, ANN001
        # We don't actually need to use the requested timestamp; each set/read
        # pair just yields the next pre-built frame in order.
        return True

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def release(self):  # noqa: D401
        return None


def _patch_capture(frames: List[np.ndarray]):
    """Context manager — patches cv2.VideoCapture in video_utils to return
    a _FakeCapture pre-loaded with `frames`."""
    return patch("src.video_utils.cv2.VideoCapture", return_value=_FakeCapture(frames))


# --- happy-path: phantom dropped, real face kept --------------------------


class TestRejectStaticFaceClusters:
    def test_static_patch_is_dropped(self):
        """A cluster sitting on a pixel-identical patch across all sampled
        frames is the canonical phantom (painted face, framed photo). Its
        max pairwise mean-abs-diff is 0 — should be rejected."""
        # Static patch at (50, 50) → (70, 70), value 200 in every frame.
        static_region = (50, 50, 70, 70, 200)
        frames = [
            _frame_with_patches(static_regions=[static_region]) for _ in range(4)
        ]
        # Cluster: center (60, 60), area=400 → side=20 (matches the patch).
        clusters = [(60, 60, 400, 0.95)]
        with _patch_capture(frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        # All-static + threshold>0 means the gate would normally drop every
        # cluster, but the "fall back to input if everything's rejected"
        # safety net kicks in to avoid the no-faces path. Verify the
        # fallback returns the input unchanged.
        assert result == clusters, (
            "All-rejected fallback should return input unchanged to avoid "
            "the no-faces downstream path."
        )

    def test_moving_patch_is_kept(self):
        """A cluster on a patch whose pixel values change between frames
        (mouth opens, head shifts) is a real face. Max pairwise mean-abs-
        diff is high (40+) — should be kept."""
        # Same region, but pixel value cycles 0 → 80 → 160 → 240.
        frames = [
            _frame_with_patches(moving_regions=[(50, 50, 70, 70, v)])
            for v in (0, 80, 160, 240)
        ]
        clusters = [(60, 60, 400, 0.95)]
        with _patch_capture(frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        assert result == clusters

    def test_keeps_real_drops_phantom_in_mixed_shot(self):
        """The motivating production case (ENG-5807 clip 1 shot 4):
        one real speaker plus two phantoms in the same shot. The gate
        must keep the speaker and drop both phantoms — NOT trigger the
        all-rejected fallback."""
        # Real speaker at (60, 60), area=400 — moving.
        # Phantom A at (160, 60), area=400 — static value 90.
        # Phantom B at (60, 160), area=400 — static value 220.
        frames = []
        for speaker_value in (0, 80, 160, 240):
            frames.append(
                _frame_with_patches(
                    moving_regions=[(50, 50, 70, 70, speaker_value)],
                    static_regions=[
                        (150, 50, 170, 70, 90),
                        (50, 150, 70, 170, 220),
                    ],
                )
            )
        clusters = [
            (60, 60, 400, 0.92),   # real speaker
            (160, 60, 400, 0.96),  # painted face — high confidence, no motion
            (60, 160, 400, 0.77),  # framed photo
        ]
        with _patch_capture(frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        assert len(result) == 1, f"Expected exactly the real speaker; got {result}"
        assert result[0][0] == 60 and result[0][1] == 60

    def test_high_confidence_does_not_save_a_phantom(self):
        """The painted-face problem specifically: the DNN detector can be
        VERY confident (conf=0.96 observed in production) on a 2D
        reproduction of a face. The motion gate must ignore confidence
        and decide purely on motion — otherwise it just rebuilds the
        same bug we're fixing."""
        frames = [
            _frame_with_patches(static_regions=[(50, 50, 70, 70, 200)])
            for _ in range(4)
        ]
        # One phantom (high conf) plus one real moving face elsewhere so
        # the fallback doesn't kick in.
        moving_frames = []
        for v in (10, 90, 170, 250):
            moving_frames.append(
                _frame_with_patches(
                    static_regions=[(50, 50, 70, 70, 200)],
                    moving_regions=[(120, 120, 140, 140, v)],
                )
            )
        clusters = [
            (60, 60, 400, 0.99),    # phantom with maximum confidence
            (130, 130, 400, 0.80),  # real face
        ]
        with _patch_capture(moving_frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        assert len(result) == 1
        assert result[0][0] == 130, "high-confidence phantom must be dropped"


# --- threshold + sample-count knobs ---------------------------------------


class TestRejectStaticFaceClustersKnobs:
    def test_lower_threshold_keeps_more(self):
        """A very low threshold (0.5) keeps clusters with small but real
        differences that the production threshold (2.0) would drop."""
        # Subtle motion: values cycle 100 → 101 → 102 → 103 → max diff ≈ 1.
        frames = [
            _frame_with_patches(moving_regions=[(50, 50, 70, 70, v)])
            for v in (100, 101, 102, 103)
        ]
        clusters_with_companion = [
            (60, 60, 400, 0.9),     # subtle-motion cluster under test
            (130, 130, 400, 0.9),   # companion so fallback doesn't fire
        ]
        # Add a clearly-moving companion patch to the frames.
        for f, v in zip(frames, (0, 80, 160, 240)):
            f[120:140, 120:140] = v

        with _patch_capture(frames):
            result_strict = reject_static_face_clusters(
                clusters_with_companion, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        assert all(c[0] != 60 for c in result_strict), (
            "subtle-motion cluster should be dropped at threshold=2.0"
        )

        # Same frames, threshold=0.5 → the subtle-motion cluster survives.
        # Rebuild frames because the prior patch loop mutated them.
        frames2 = [
            _frame_with_patches(moving_regions=[(50, 50, 70, 70, v)])
            for v in (100, 101, 102, 103)
        ]
        for f, v in zip(frames2, (0, 80, 160, 240)):
            f[120:140, 120:140] = v
        with _patch_capture(frames2):
            result_loose = reject_static_face_clusters(
                clusters_with_companion, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=0.5, samples=4,
            )
        assert any(c[0] == 60 for c in result_loose), (
            "subtle-motion cluster should survive at threshold=0.5"
        )

    def test_samples_below_two_skips_gate(self):
        """The gate needs at least 2 frames to compute any diff. If a caller
        asks for fewer, return clusters unchanged rather than dropping all."""
        clusters = [(60, 60, 400, 0.9)]
        with _patch_capture([]):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=1,
            )
        assert result == clusters


# --- defensive paths ------------------------------------------------------


class TestRejectStaticFaceClustersDefensive:
    def test_empty_input_returns_empty(self):
        result = reject_static_face_clusters(
            [], Path("/dummy.mp4"), 0.0, 4.0,
        )
        assert result == []

    def test_zero_duration_returns_unchanged(self):
        clusters = [(60, 60, 400, 0.9)]
        result = reject_static_face_clusters(
            clusters, Path("/dummy.mp4"), 5.0, 5.0,
        )
        assert result == clusters

    def test_unopenable_video_returns_unchanged(self):
        clusters = [(60, 60, 400, 0.9)]

        class _UnopenableCapture:
            def isOpened(self):
                return False
            def release(self):
                pass

        with patch(
            "src.video_utils.cv2.VideoCapture",
            return_value=_UnopenableCapture(),
        ):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
            )
        # Cannot evaluate motion → don't silently drop everything; pass
        # through so downstream code sees the same input it would have.
        assert result == clusters

    def test_too_few_readable_frames_returns_unchanged(self):
        """If only 1 frame is readable (e.g. a corrupt video), the gate
        cannot compute any pairwise diff. Pass through rather than drop."""
        clusters = [(60, 60, 400, 0.9)]
        # Only 1 frame in the buffer — every subsequent read() returns False.
        one_frame = [_frame_with_patches()]
        with _patch_capture(one_frame):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0, samples=4,
            )
        assert result == clusters

    def test_all_rejected_falls_back_to_input(self):
        """If every cluster looks static (e.g. an entire freeze-frame
        shot, or a real-but-still listening shot), fall back to the
        input rather than handing downstream code an empty list. Empty
        face_centers triggers the no-faces fallback path which can
        produce a worse crop than 'wrong but plausible'."""
        # All-static frame; cluster will look static.
        frames = [_frame_with_patches(static_regions=[(50, 50, 70, 70, 200)])] * 4
        clusters = [(60, 60, 400, 0.9)]
        with _patch_capture(frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        assert result == clusters

    def test_tiny_bbox_is_kept(self):
        """A cluster whose reconstructed bbox is <10×10 (degenerate
        sqrt(area) for a near-zero area) is kept rather than evaluated —
        the patch is too small to measure motion reliably and dropping
        it would silently remove real but distant faces."""
        # area=25 → side=5 → bbox <10px. Padding adds 1 each side → 7px.
        frames = [_frame_with_patches() for _ in range(4)]
        clusters = [(60, 60, 25, 0.9)]
        with _patch_capture(frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 4.0,
                motion_threshold=2.0, samples=4,
            )
        assert result == clusters


# --- production fixture ---------------------------------------------------


class TestRejectStaticFaceClustersProductionFixture:
    """Replays the cluster shape from BN job 2623b681 clip 1 shot 4 —
    the most clear-cut production case the fix targets. xs=[238, 907, 1661]
    pre-gate; the gate is expected to leave only x=907 (the real speaker)."""

    def test_clip1_shot4_keeps_only_real_speaker(self):
        # Build a 1920×1080 frame; populate three regions matching the
        # production cluster positions. The two phantoms are static, the
        # centre speaker moves.
        H, W = 1080, 1920
        def base(speaker_value: int) -> np.ndarray:
            f = np.full((H, W, 3), 90, dtype=np.uint8)
            # Phantom 1: painted face on left side. area=18445 → side≈136.
            f[238:374, 170:306] = 60
            # Phantom 2: framed photo on right side. area=12980 → side≈114.
            f[298:412, 1604:1718] = 200
            # Real speaker: area=32984 → side≈182. Centred around (907, 237).
            f[146:328, 816:998] = speaker_value
            return f

        frames = [base(v) for v in (10, 90, 170, 250)]
        clusters = [
            (238, 306, 18445, 0.96),   # painted face — high conf, no motion
            (907, 237, 32984, 0.92),   # real speaker
            (1661, 354, 12980, 0.77),  # framed photo
        ]
        with _patch_capture(frames):
            result = reject_static_face_clusters(
                clusters, Path("/dummy.mp4"), 0.0, 5.5,
                motion_threshold=2.0, samples=4,
            )
        assert len(result) == 1, (
            f"Expected only the real speaker to survive; got {result}"
        )
        assert result[0][0] == 907 and result[0][1] == 237
