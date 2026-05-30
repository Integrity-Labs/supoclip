"""
Tests for the crop-overlay diagnostic script (ENG-5719).

The script lives at backend/scripts/crop_overlay.py so it can run standalone
without importing supoclip's full dependency tree. These tests load it as a
module to exercise the pure filtergraph builder + escape helpers.
"""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def crop_overlay():
    """Load the standalone script as a module so we can test its functions."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "crop_overlay.py"
    spec = importlib.util.spec_from_file_location("crop_overlay", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_plan() -> dict:
    """Minimal plan covering all four overlay element types: crop windows,
    scene cuts, face dots, per-shot labels. Numbers chosen to be obvious
    in the rendered filtergraph string."""
    return {
        "schema_version": 1,
        "mode": "cut",
        "source_width": 1920,
        "source_height": 1080,
        "source_duration": 48.2,
        "crop_w": 608,
        "crop_h": 1080,
        "scene_cut_times": [8.6, 14.3],
        "shots": [
            {
                "shot_index": 0,
                "start": 0.1,
                "end": 8.6,
                "face_count": 2,
                "face_centers": [
                    {"center_x": 700, "center_y": 540, "area": 12345, "confidence": 0.91},
                    {"center_x": 1500, "center_y": 520, "area": 11000, "confidence": 0.85},
                ],
                "mode": "single-frame",
                "x_target": 940,
            },
            {
                "shot_index": 1,
                "start": 8.6,
                "end": 14.3,
                "face_count": 1,
                "face_centers": [
                    {"center_x": 600, "center_y": 530, "area": 13000, "confidence": 0.88}
                ],
                "mode": "single-frame",
                "x_target": 574,
            },
        ],
        "merged_timeline": [
            {"start": 0.1, "end": 8.6, "x": 940},
            {"start": 8.6, "end": 14.3, "x": 574},
        ],
    }


class TestFiltergraphBuilder:
    def test_emits_one_crop_box_per_merged_segment(self, crop_overlay):
        plan = _make_plan()
        fg = crop_overlay.build_filtergraph(plan)
        # Two merged segments → two crop drawbox calls. The other drawbox
        # calls (scene cuts, face dots) use thickness=fill, the crop boxes
        # use a specific outline thickness — count those.
        crop_box_count = fg.count(f"thickness={crop_overlay._CROP_BOX_THICKNESS}")
        assert crop_box_count == 2, fg

    def test_emits_scene_cut_lines(self, crop_overlay):
        plan = _make_plan()
        fg = crop_overlay.build_filtergraph(plan)
        # Two scene cuts → two cyan vertical lines. Match on the colour
        # rather than counting drawbox calls (face dots also use drawbox).
        assert fg.count(f"color={crop_overlay._SCENE_CUT_COLOR}") == 2, fg

    def test_emits_face_dot_per_detection(self, crop_overlay):
        plan = _make_plan()
        fg = crop_overlay.build_filtergraph(plan)
        # 2 detections in shot 0 + 1 in shot 1 = 3 red dots.
        assert fg.count(f"color={crop_overlay._FACE_DOT_COLOR}") == 3, fg

    def test_emits_drawtext_label_per_shot(self, crop_overlay):
        plan = _make_plan()
        fg = crop_overlay.build_filtergraph(plan)
        assert fg.count("drawtext=") == 2, fg
        assert "shot 0 single-frame faces=2 x=940" in fg
        assert "shot 1 single-frame faces=1 x=574" in fg

    def test_uses_enable_between_for_time_scoping(self, crop_overlay):
        plan = _make_plan()
        fg = crop_overlay.build_filtergraph(plan)
        # Every overlay element is time-scoped — there should be at least
        # one between(t,...) per element. Cheap heuristic: count enable=
        # against expected total.
        # 2 crop + 2 cuts + 3 face dots + 2 labels = 9 elements.
        assert fg.count("enable=") == 9, fg

    def test_handles_no_faces_shot_label(self, crop_overlay):
        plan = _make_plan()
        plan["shots"].append(
            {
                "shot_index": 2,
                "start": 14.3,
                "end": 20.0,
                "face_count": 0,
                "face_centers": [],
                "mode": "no-faces",
                "x_target": None,
            }
        )
        fg = crop_overlay.build_filtergraph(plan)
        # No-faces shot label should read x=n/a, not x=None.
        assert "shot 2 no-faces faces=0 x=n/a" in fg

    def test_clamps_face_dot_to_frame_origin(self, crop_overlay):
        """Detector can return centers near the edge; the drawbox top-left
        must not go negative or ffmpeg refuses to parse it."""
        plan = _make_plan()
        plan["shots"][0]["face_centers"] = [
            {"center_x": 3, "center_y": 4, "area": 100, "confidence": 0.6}
        ]
        plan["shots"][1]["face_centers"] = []
        fg = crop_overlay.build_filtergraph(plan)
        # The face dot for (3, 4) at radius 8 would otherwise start at
        # x=-5, y=-4 — verify we clamped to 0.
        assert "drawbox=x=0:y=0:w=16:h=16" in fg, fg


class TestEscapeHelpers:
    def test_ff_str_escapes_backslash_and_colon(self, crop_overlay):
        assert crop_overlay._ff_str("a:b") == "a\\:b"
        assert crop_overlay._ff_str("c\\d") == "c\\\\d"

    def test_ff_time_trims_trailing_zeros(self, crop_overlay):
        assert crop_overlay._ff_time(1.0) == "1"
        assert crop_overlay._ff_time(1.500) == "1.5"
        assert crop_overlay._ff_time(8.612) == "8.612"


class TestReframeSidecarUriFromClip:
    """ENG-5719 Phase 2: the endpoint derives the sidecar's storage URI
    from the clip's storage URI via os.path.splitext. Path.with_suffix
    can't be used because it mangles s3:// URIs (treats s3: as a path
    component)."""

    def test_local_path(self):
        from src.api.routes.tasks import _reframe_sidecar_uri_from_clip

        assert (
            _reframe_sidecar_uri_from_clip("/tmp/clips/abc.mp4")
            == "/tmp/clips/abc.reframe_plan.json"
        )

    def test_s3_uri(self):
        from src.api.routes.tasks import _reframe_sidecar_uri_from_clip

        assert (
            _reframe_sidecar_uri_from_clip("s3://bucket/clips/abc.mp4")
            == "s3://bucket/clips/abc.reframe_plan.json"
        )

    def test_multi_dot_basename(self):
        """Only the LAST .ext gets swapped so versioned filenames work."""
        from src.api.routes.tasks import _reframe_sidecar_uri_from_clip

        assert (
            _reframe_sidecar_uri_from_clip("s3://bucket/clip.v2.mp4")
            == "s3://bucket/clip.v2.reframe_plan.json"
        )

    def test_no_extension(self):
        """Edge case: clip URI without an extension — append the sidecar
        suffix rather than corrupting the basename."""
        from src.api.routes.tasks import _reframe_sidecar_uri_from_clip

        assert (
            _reframe_sidecar_uri_from_clip("s3://bucket/abc")
            == "s3://bucket/abc.reframe_plan.json"
        )


class TestPlanLoader:
    def test_loads_v1_plan_from_file(self, crop_overlay, tmp_path):
        plan = _make_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))
        loaded = crop_overlay.load_plan_from_file(plan_path)
        assert loaded["schema_version"] == 1
        assert loaded["crop_w"] == 608

    def test_warns_on_unknown_schema_version(self, crop_overlay, tmp_path, caplog):
        plan = _make_plan()
        plan["schema_version"] = 999
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))
        with caplog.at_level("WARNING", logger="crop_overlay"):
            crop_overlay.load_plan_from_file(plan_path)
        assert any("schema_version=999" in rec.message for rec in caplog.records)
