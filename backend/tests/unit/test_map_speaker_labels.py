"""
Tests for map_speaker_labels_to_sides + score_speaker_labels_by_side
(ENG-5755).

Symptom on prod (BN job d5594aa6 clip_2_1018-1059 sidecar): every L+R
shot fell to single-frame with fallback_reason='single_side_speakers'.
Both diarized speakers were collapsing onto the same side because the
losing label never accumulated any score — its utterance windows
didn't overlap any motion sample time, so it dropped out of the
sorted list entirely and ordered[-1] aliased back to ordered[0].

The fix initialises every label with airtime in the stats dict
(defaulting to leftness=0), so two distinct labels always map to two
distinct sides — by direct correlation when motion is present, and by
elimination (force-pair the only label to "left" and the other to
"right") when one label has no overlapping samples.
"""

import pytest

from src.video_utils import (
    map_speaker_labels_to_sides,
    score_speaker_labels_by_side,
)


def _flat_motion_left_heavy(n: int = 100):
    """Realistic motion frames: left zone consistently active, right zone idle."""
    times = [i * 0.1 for i in range(n)]
    left = [3.0 + (i % 3) * 0.1 for i in range(n)]
    right = [0.2 + (i % 2) * 0.05 for i in range(n)]
    return times, left, right


class TestScoreSpeakerLabelsByPide:
    def test_returns_left_right_leftness_and_airtime_per_label(self):
        times, left, right = _flat_motion_left_heavy()
        stats = score_speaker_labels_by_side(
            utterances=[
                {"start": 0.0, "end": 3.0, "speaker": "A"},
                {"start": 4.0, "end": 7.0, "speaker": "B"},
            ],
            times=times, left_values=left, right_values=right,
        )
        assert set(stats.keys()) == {"A", "B"}
        for label in ("A", "B"):
            assert set(stats[label].keys()) == {"left", "right", "leftness", "airtime"}
            assert stats[label]["leftness"] == pytest.approx(
                stats[label]["left"] - stats[label]["right"]
            )
            assert stats[label]["airtime"] == pytest.approx(3.0)

    def test_label_with_no_overlapping_motion_samples_still_appears(self):
        """ENG-5755 root cause: if a label's utterances are entirely between
        sampled frames, it accumulates airtime but never any (left, right)
        score. The stats dict must still surface the label with zeroed
        motion totals — otherwise ``map_speaker_labels_to_sides`` drops it
        from ``ordered`` and the two-label case collapses."""
        # Sparse motion samples at t=0.0 and t=10.0 — speaker B's 4.5-5.0s
        # window falls entirely between samples.
        stats = score_speaker_labels_by_side(
            utterances=[
                {"start": 0.0, "end": 1.0, "speaker": "A"},
                {"start": 4.5, "end": 5.0, "speaker": "B"},
            ],
            times=[0.0, 10.0],
            left_values=[5.0, 5.0],
            right_values=[1.0, 1.0],
        )
        assert "B" in stats, (
            "Label with airtime but no overlapping samples must still appear "
            "in stats — that's the whole point of the ENG-5755 fix."
        )
        assert stats["B"]["airtime"] > 0
        assert stats["B"]["left"] == 0.0
        assert stats["B"]["right"] == 0.0
        assert stats["B"]["leftness"] == 0.0

    def test_empty_inputs_return_empty(self):
        assert score_speaker_labels_by_side([], [], [], []) == {}
        assert score_speaker_labels_by_side(
            [{"start": 0, "end": 1, "speaker": "A"}], [], [], []
        ) == {}
        # Mismatched left/right lengths — defensive guard.
        assert score_speaker_labels_by_side(
            [{"start": 0, "end": 1, "speaker": "A"}],
            [0.0, 0.1], [1.0], [2.0, 2.5],
        ) == {}


class TestMapSpeakerLabelsToSides:
    def test_two_labels_with_opposite_motion_map_to_opposite_sides(self):
        """Happy path — A speaks while left zone has motion, B while right does."""
        times = [i * 0.1 for i in range(60)]
        # A talks during left-active frames, B talks during right-active frames.
        left = [5.0] * 30 + [0.2] * 30
        right = [0.2] * 30 + [5.0] * 30
        mapping = map_speaker_labels_to_sides(
            utterances=[
                {"start": 0.0, "end": 3.0, "speaker": "A"},
                {"start": 3.0, "end": 6.0, "speaker": "B"},
            ],
            times=times, left_values=left, right_values=right,
        )
        assert mapping == {"A": "left", "B": "right"}

    def test_one_label_without_overlapping_samples_still_yields_two_sides(self):
        """ENG-5755 regression: when only one of two labels accumulates a
        motion score, the other must be force-paired to the opposite side
        — not dropped from the mapping (which collapses the caller's set
        check to len=1 and bails the entire two-shot-cut path)."""
        # B's utterance (4.5-5.0s) falls between motion samples at 0.0 and 10.0
        # — historically B never appeared in the scores dict at all.
        mapping = map_speaker_labels_to_sides(
            utterances=[
                {"start": 0.0, "end": 1.0, "speaker": "A"},
                {"start": 4.5, "end": 5.0, "speaker": "B"},
            ],
            times=[0.0, 10.0],
            left_values=[5.0, 5.0],
            right_values=[1.0, 1.0],
        )
        assert set(mapping.keys()) == {"A", "B"}, (
            f"Both labels must appear in mapping; got {mapping}"
        )
        assert set(mapping.values()) == {"left", "right"}, (
            "Two distinct labels must yield two distinct sides — the whole "
            f"point of the ENG-5755 fix. Got {mapping}."
        )
        # A had the only real motion sample (heavy left) — A must land on left.
        assert mapping["A"] == "left"
        assert mapping["B"] == "right"

    def test_three_labels_extremes_force_paired_middle_follows_sign(self):
        """Backwards-compatibility: with 3+ labels, the most-left and
        most-right are still force-paired to their sides; the middle label
        follows its own leftness sign."""
        times = [i * 0.1 for i in range(60)]
        left = [5.0] * 60
        right = [0.5] * 60  # heavily left-biased throughout
        mapping = map_speaker_labels_to_sides(
            utterances=[
                {"start": 0.0, "end": 2.0, "speaker": "A"},  # left during left-heavy
                {"start": 2.0, "end": 4.0, "speaker": "B"},  # middle, also left-heavy
                {"start": 4.0, "end": 6.0, "speaker": "C"},  # last
            ],
            times=times, left_values=left, right_values=right,
        )
        # All three labels appear; left/right both present.
        assert set(mapping.keys()) == {"A", "B", "C"}
        assert {"left", "right"} <= set(mapping.values())

    def test_empty_utterances_returns_empty_mapping(self):
        assert map_speaker_labels_to_sides([], [0.0], [1.0], [1.0]) == {}
