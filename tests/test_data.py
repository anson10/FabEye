"""Tests for data generation and loading."""

import json
import pytest
import tempfile
from pathlib import Path


class TestWaferDataGenerator:
    def test_generates_correct_count(self):
        from data.generator import WaferDataGenerator
        gen = WaferDataGenerator(n_wafers=50, seed=0)
        samples = gen.generate()
        assert len(samples) == 50

    def test_sample_structure(self):
        from data.generator import WaferDataGenerator, PROCESS_STEPS
        gen = WaferDataGenerator(n_wafers=5, seed=1)
        samples = gen.generate()
        s = samples[0]

        assert "wafer_id" in s
        assert "process_steps" in s
        assert "defect" in s
        assert "node_features" in s
        assert "adjacency" in s
        assert len(s["process_steps"]) == len(PROCESS_STEPS)

    def test_defect_types_valid(self):
        from data.generator import WaferDataGenerator, DEFECT_TYPES
        gen = WaferDataGenerator(n_wafers=100, seed=2)
        samples = gen.generate()
        valid_types = set(DEFECT_TYPES.keys())
        for s in samples:
            assert s["defect"]["defect_type"] in valid_types

    def test_defect_distribution_not_all_none(self):
        """Dataset should have some defective wafers."""
        from data.generator import WaferDataGenerator
        gen = WaferDataGenerator(n_wafers=200, seed=3)
        samples = gen.generate()
        has_defect = sum(1 for s in samples if s["defect"]["has_defect"])
        assert has_defect > 10, f"Too few defective wafers: {has_defect}"

    def test_node_features_normalized(self):
        """All node features should be in [0, 1]."""
        from data.generator import WaferDataGenerator
        gen = WaferDataGenerator(n_wafers=20, seed=4)
        samples = gen.generate()
        for s in samples:
            for step_feats in s["node_features"]:
                for val in step_feats:
                    assert 0.0 <= val <= 1.0, f"Feature out of range: {val}"

    def test_save_creates_file(self, tmp_path):
        from data.generator import WaferDataGenerator
        out = str(tmp_path / "test_wafers.json")
        gen = WaferDataGenerator(n_wafers=10, seed=5)
        gen.save(out)
        assert Path(out).exists()
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 10

    def test_adjacency_structure(self):
        """Adjacency should form a bidirectional linear chain."""
        from data.generator import WaferDataGenerator, PROCESS_STEPS
        gen = WaferDataGenerator(n_wafers=5, seed=6)
        samples = gen.generate()
        n_steps = len(PROCESS_STEPS)
        expected_edges = 2 * (n_steps - 1)
        for s in samples:
            assert len(s["adjacency"]) == expected_edges

    def test_reproducibility(self):
        from data.generator import WaferDataGenerator
        s1 = WaferDataGenerator(n_wafers=10, seed=99).generate()
        s2 = WaferDataGenerator(n_wafers=10, seed=99).generate()
        assert s1[0]["defect"] == s2[0]["defect"]
