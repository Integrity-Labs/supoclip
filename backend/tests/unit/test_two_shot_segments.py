"""
Tests for build_two_shot_segments + the speaker-aware single-frame
fallback in build_per_shot_cut_plan (ENG-5751).

Uses real production data captured from BN job
d5594aa6-5a82-4e82-9955-aabc7f4c890b (supoclip task
85616068-0abf-4ca5-8561-8aba17ad5c48, clip 2 shot 1) as a fixture so
the regression test fires off the actual layout that broke framing:
a stable L+R two-shot where build_two_shot_segments returned None,
and pick_shot_crop_x deterministically picked the leftmost (heavier)
face for x_target=0 — cropping the right speaker out for ~10s.

The two-shot-cut path can't always trigger (diarization, lip-motion,
or label_to_side may all bail). The fallback added in ENG-5751 is the
safety net: whenever lip motion ran, the single-frame fallback uses it
to frame the active speaker — not the leftmost face.
"""

from pathlib import Path

import pytest

from src import video_utils
from src.video_utils import build_two_shot_segments


# ENG-5751 clip 2 shot 1 (0.2-9.6s). 1920x1080 source, 606-wide vertical crop.
# After ENG-5719 Phase 3 dedupe, exactly 2 face_centers per shot:
#   left  (239, 303) area=21000 conf=0.98  — well-lit foreground host
#   right (1639, 329) area=12200 conf=0.59 — secondary host, smaller in frame
# The left face's area*confidence (20580) dominates the right's (7198), so
# pick_shot_crop_x always picks x_target=0 (leftmost) — even when the right
# host is the one talking. This is the user-visible bug.
SHOT_1_REGIONS = {
    "left": {
        "center_x": 239, "center_y": 303,
        "roi_x": 200, "roi_y": 280, "roi_w": 160, "roi_h": 140,
        "tile_x": 80, "tile_y": 180, "tile_w": 320, "tile_h": 260,
    },
    "right": {
        "center_x": 1639, "center_y": 329,
        "roi_x": 1580, "roi_y": 300, "roi_w": 160, "roi_h": 140,
        "tile_x": 1480, "tile_y": 200, "tile_w": 320, "tile_h": 260,
    },
}
SHOT_1_START = 0.2
SHOT_1_END = 9.64
WIDTH = 1920
HEIGHT = 1080
CROP_W = 606


class TestBuildTwoShotSegmentsResultShape:
    """ENG-5751 changed the return type from Optional[List] to a dict carrying
    segments + fallback diagnostics. Lock the new shape down."""

    def test_returns_dict_with_required_keys(self, monkeypatch, tmp_path):
        # Right speaker is talking — give them more lip motion.
        monkeypatch.setattr(
            video_utils, "measure_region_motion",
            lambda *a, **kw: ([0.0, 1.0, 2.0], [0.1, 0.1, 0.1], [4.0, 5.0, 6.0]),
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH, utterances=[],
        )
        assert set(result.keys()) >= {"segments", "fallback_reason", "dominant_side"}


class TestSpeakerAwareFallbackOnStableLRShot:
    """The user-visible ENG-5751 bug: when build_two_shot_segments can't build a
    cut timeline on a stable L+R shot, the dominant_side from lip motion must
    still be returned so the caller can frame the active speaker instead of
    defaulting to leftmost."""

    def test_no_utterances_falls_back_to_motion_dominant_right(self, monkeypatch, tmp_path):
        """No diarization overlap → motion picks the speaker. Right wins."""
        monkeypatch.setattr(
            video_utils, "measure_region_motion",
            # Right has ~5x more lip motion than left.
            lambda *a, **kw: ([0.0, 1.0, 2.0], [0.2, 0.1, 0.2], [3.0, 4.0, 5.0]),
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH, utterances=[],
        )
        assert result["segments"] is None
        assert result["fallback_reason"] == "no_utterances"
        assert result["dominant_side"] == "right", (
            "Right speaker has 5x left's motion — fallback must surface right "
            "so caller can frame the active speaker (not the leftmost face)."
        )

    def test_no_utterances_falls_back_to_motion_dominant_left(self, monkeypatch, tmp_path):
        """Mirror case: when the left host is talking, fallback frames left."""
        monkeypatch.setattr(
            video_utils, "measure_region_motion",
            lambda *a, **kw: ([0.0, 1.0, 2.0], [5.0, 4.0, 5.0], [0.1, 0.2, 0.1]),
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH, utterances=[],
        )
        assert result["segments"] is None
        assert result["dominant_side"] == "left"

    def test_no_utterances_and_motion_unavailable_is_no_rescue(
        self, monkeypatch, tmp_path
    ):
        """When ffmpeg motion fails too, we genuinely have no signal —
        caller will fall through to pick_shot_crop_x's geometric pick."""
        monkeypatch.setattr(
            video_utils, "measure_region_motion", lambda *a, **kw: None
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH, utterances=[],
        )
        assert result["segments"] is None
        assert result["fallback_reason"] == "no_utterances_motion_unavailable"
        assert result["dominant_side"] is None

    def test_motion_unavailable_with_utterances(self, monkeypatch, tmp_path):
        """Diarization overlap but ffmpeg motion fails — no rescue side."""
        monkeypatch.setattr(
            video_utils, "measure_region_motion", lambda *a, **kw: None
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH,
            utterances=[
                {"start": 1.0, "end": 4.0, "speaker": "A"},
                {"start": 5.0, "end": 8.0, "speaker": "B"},
            ],
        )
        assert result["segments"] is None
        assert result["fallback_reason"] == "motion_unavailable"
        assert result["dominant_side"] is None

    def test_single_side_speakers_returns_dominant(self, monkeypatch, tmp_path):
        """Two diarized speakers but label_to_side maps both to the same
        side — historically returned None. Now we still return the
        motion-derived dominant side so the fallback can use it."""
        monkeypatch.setattr(
            video_utils, "measure_region_motion",
            lambda *a, **kw: ([0.0, 1.0, 2.0], [0.1, 0.1, 0.1], [3.0, 4.0, 5.0]),
        )
        monkeypatch.setattr(
            video_utils, "map_speaker_labels_to_sides",
            lambda *a, **kw: {"A": "right", "B": "right"},
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH,
            utterances=[
                {"start": 1.0, "end": 4.0, "speaker": "A"},
                {"start": 5.0, "end": 8.0, "speaker": "B"},
            ],
        )
        assert result["segments"] is None
        assert result["fallback_reason"] == "single_side_speakers"
        assert result["dominant_side"] == "right"


class TestSuccessfulTwoShotCutStillWorks:
    """Don't regress the happy path: when both speakers map to distinct
    sides and the timeline has ≥2 segments, we still get a clip-relative
    {start, end, x} list and no fallback_reason."""

    def test_happy_path_returns_segments_and_no_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            video_utils, "measure_region_motion",
            lambda *a, **kw: ([0.0, 1.0, 2.0], [3.0, 0.1, 3.0], [0.1, 4.0, 0.1]),
        )
        monkeypatch.setattr(
            video_utils, "map_speaker_labels_to_sides",
            lambda *a, **kw: {"A": "left", "B": "right"},
        )
        result = build_two_shot_segments(
            tmp_path / "clip.mp4", SHOT_1_START, SHOT_1_END,
            SHOT_1_REGIONS, CROP_W, WIDTH,
            utterances=[
                {"start": 1.0, "end": 4.0, "speaker": "A"},
                {"start": 5.0, "end": 8.0, "speaker": "B"},
            ],
        )
        assert result["fallback_reason"] is None
        assert result["segments"] is not None
        assert len(result["segments"]) >= 2
        # Segments should be clip-relative and frame the actual L/R positions.
        for seg in result["segments"]:
            assert seg["start"] >= SHOT_1_START - 1e-6
            assert seg["end"] <= SHOT_1_END + 1e-6
            # x must be one of the two clamped crop offsets (left or right).
            assert seg["x"] in {0, 1314}  # 1639 - 606//2 = 1336 → clamped to 1314
