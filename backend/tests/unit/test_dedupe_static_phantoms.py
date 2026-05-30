"""
Tests for dedupe_static_phantoms (ENG-5719 Phase 3).

Uses real production data captured from job 9c4026d1's reframe sidecar as
a fixture, so the regression test fires off the actual pattern that motivated
the function.
"""

import pytest

from src.video_utils import dedupe_static_phantoms


# Realistic test fixtures captured from job
# 9c4026d1-2df1-421f-94ba-59a80e554c5f, supoclip task
# 1cf245e9-9172-4ce1-8a9c-625d96870843, clip 1
# (s3://supoclipstack-supoclipclipsbucket.../clips/clip_1_1018-1105_275a2c44704c.reframe_plan.json
# shot 8 raw face_centers, pre-dedupe). The 5-detection cluster at the
# left edge is the actual left speaker; the 12-detection cluster at
# (1105, 597) is the phantom (a static piece of furniture the DNN keeps
# misclassifying) that broke the crop_x selection before this fix.
SHOT_8_RAW_DETECTIONS = [
    # 5 real detections of the left speaker — note the inter-frame
    # positional spread of 13-20px on x and 27px on y (real movement)
    (87, 435, 22578, 0.96),
    (99, 427, 22646, 0.84),
    (126, 430, 20066, 0.98),
    (143, 413, 20000, 0.96),
    (156, 408, 24700, 0.81),
    # 12 phantom detections at (1105, 597) ±2 — static object, ~zero
    # positional variance (the giveaway). Lower-but-still-passing confidence.
    (1104, 598, 18490, 0.62),
    (1105, 598, 17974, 0.71),
    (1105, 597, 17888, 0.73),
    (1105, 597, 18444, 0.71),
    (1105, 598, 17935, 0.71),
    (1106, 597, 17680, 0.73),
    (1106, 597, 18183, 0.74),
    (1106, 597, 18270, 0.74),
    (1106, 597, 18270, 0.74),
    (1106, 597, 18480, 0.76),
    (1106, 597, 18444, 0.77),
    (1107, 597, 18357, 0.77),
]


class TestDedupeStaticPhantoms:
    def test_collapses_shot_8_to_two_clusters(self):
        """The motivating case: 17 raw detections → 2 distinct entities.
        The phantom should NOT outvote the real face after dedupe."""
        result = dedupe_static_phantoms(SHOT_8_RAW_DETECTIONS)
        assert len(result) == 2, (
            f"Expected exactly 2 clusters (1 real speaker + 1 phantom); "
            f"got {len(result)}: {result}"
        )

    def test_preserves_real_face_position(self):
        """The left speaker cluster (5 detections, x≈87-156) should
        collapse to a representative near the median of those 5, not be
        pulled toward the phantom."""
        result = dedupe_static_phantoms(SHOT_8_RAW_DETECTIONS)
        left_speaker = next(r for r in result if r[0] < 500)
        # Median of [87, 99, 126, 143, 156] = 126; median y of
        # [435, 427, 430, 413, 408] = 427. Tolerate small variation.
        assert 80 <= left_speaker[0] <= 170, left_speaker
        assert 400 <= left_speaker[1] <= 450, left_speaker
        # Confidence should be the mean of the cluster's confidences (~0.91)
        # — higher than ANY phantom detection (max 0.77). Downstream code
        # can use this to weight clusters.
        assert left_speaker[3] > 0.85, f"left speaker confidence too low: {left_speaker[3]}"

    def test_phantom_collapses_to_one_vote(self):
        """The phantom's 12 raw detections should collapse to a SINGLE
        representative. This is the whole point of the fix — without it,
        the phantom carries 12 votes through downstream clustering."""
        result = dedupe_static_phantoms(SHOT_8_RAW_DETECTIONS)
        phantom = next(r for r in result if r[0] > 1000)
        # Position should be near (1105, 597) — median of the 12 raw
        # samples. Confidence is the average of the 12 raw confidences
        # (0.62-0.77), which is < 0.85 — distinguishable from real-face
        # cluster confidence.
        assert 1100 <= phantom[0] <= 1110
        assert 595 <= phantom[1] <= 600
        assert phantom[3] < 0.85

    def test_phantom_no_longer_outvotes_real_speaker(self):
        """The core regression assertion: real speaker and phantom each
        carry equal weight (1 vote each) after dedupe — not 5 vs 12."""
        result = dedupe_static_phantoms(SHOT_8_RAW_DETECTIONS)
        # Both clusters present exactly once.
        left = [r for r in result if r[0] < 500]
        phantom = [r for r in result if r[0] > 1000]
        assert len(left) == 1
        assert len(phantom) == 1


class TestDedupeStaticPhantomsEdgeCases:
    def test_empty_input_returns_empty(self):
        assert dedupe_static_phantoms([]) == []

    def test_single_detection_returns_unchanged(self):
        """No dedupe possible with 1 detection — return as-is."""
        single = [(500, 400, 20000, 0.9)]
        assert dedupe_static_phantoms(single) == single

    def test_two_distinct_speakers_stay_distinct(self):
        """Left speaker at x=200 + right speaker at x=1700 are well
        beyond `radius` apart — must remain 2 separate clusters."""
        detections = [
            (200, 400, 20000, 0.9),  # left
            (210, 410, 20000, 0.92),  # left, +10px
            (1700, 400, 22000, 0.88),  # right
            (1710, 415, 22000, 0.91),  # right, +10px
        ]
        result = dedupe_static_phantoms(detections)
        assert len(result) == 2

    def test_moving_speaker_collapses_within_radius(self):
        """A single speaker moves their head between samples but stays
        within `radius` (35px) of the cluster median — should collapse
        to ONE cluster, not split into a per-frame entity. This is the
        symmetric case to the phantom test: real motion within radius
        is fine."""
        detections = [
            (500, 400, 20000, 0.88),
            (508, 405, 20000, 0.91),
            (515, 398, 20000, 0.86),
            (522, 402, 20000, 0.89),
            (528, 410, 20000, 0.92),
        ]
        result = dedupe_static_phantoms(detections)
        assert len(result) == 1
        rep = result[0]
        # Median x of [500, 508, 515, 522, 528] = 515
        assert rep[0] == 515

    def test_three_distinct_speakers(self):
        """Some shots show 3 speakers (e.g. shot 6 in clip 1). All three
        should survive as separate clusters."""
        detections = [
            (150, 400, 20000, 0.9),  # left
            (160, 410, 20000, 0.92),  # left
            (800, 350, 22000, 0.85),  # middle
            (810, 355, 22000, 0.87),  # middle
            (1700, 400, 20000, 0.88),  # right
            (1710, 410, 20000, 0.9),  # right
        ]
        result = dedupe_static_phantoms(detections)
        assert len(result) == 3

    def test_radius_argument_is_respected(self):
        """Verifies the calibration knob is plumbed through. Empirical
        calibration used radius=80 in production; this test would catch a
        regression where someone silently changes the default."""
        detections = [
            (100, 100, 20000, 0.9),
            (200, 100, 20000, 0.9),  # 100px right — beyond radius=80 default
        ]
        # With radius=50 (tighter), they should stay separate.
        result_tight = dedupe_static_phantoms(detections, radius=50)
        assert len(result_tight) == 2
        # With radius=80 (default), they're 100px apart so STILL separate.
        result_default = dedupe_static_phantoms(detections)
        assert len(result_default) == 2
        # With radius=150 (loose), they merge.
        result_loose = dedupe_static_phantoms(detections, radius=150)
        assert len(result_loose) == 1

    def test_three_distinct_speakers_at_default_radius(self):
        """Left at x=150, middle at x=800, right at x=1700. All three
        are ≥650px apart so default radius=80 must keep them distinct
        (regression guard against the calibration moving too high)."""
        detections = [
            (150, 400, 20000, 0.9),
            (160, 410, 20000, 0.92),
            (800, 350, 22000, 0.85),
            (810, 355, 22000, 0.87),
            (1700, 400, 20000, 0.88),
            (1710, 410, 20000, 0.9),
        ]
        result = dedupe_static_phantoms(detections)
        assert len(result) == 3
