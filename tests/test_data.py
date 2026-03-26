"""Tests for data module constants and synthetic fixture integrity."""
from pitchlens.constants import TARGET_COL
from pitchlens.data.poi_metrics import (
    ALL_BIO_COLS,
    ENERGY_COLS,
    GRF_COLS,
    HP_ROM_COLS,
    HP_STRENGTH_COLS,
    KINEMATIC_COLS,
    KINETIC_COLS,
)


class TestColumnGroups:
    def test_non_empty(self):
        assert len(KINEMATIC_COLS) > 0
        assert len(KINETIC_COLS) > 0
        assert len(ENERGY_COLS) > 0
        assert len(GRF_COLS) > 0
        assert len(HP_STRENGTH_COLS) > 0
        assert len(HP_ROM_COLS) > 0

    def test_all_bio_is_union(self):
        expected = set(KINEMATIC_COLS + KINETIC_COLS + ENERGY_COLS + GRF_COLS)
        assert set(ALL_BIO_COLS) == expected

    def test_no_duplicates(self):
        for name, cols in [
            ("KINEMATIC", KINEMATIC_COLS),
            ("ENERGY", ENERGY_COLS),
            ("GRF", GRF_COLS),
        ]:
            assert len(cols) == len(set(cols)), f"Duplicates in {name}_COLS"

    def test_target_col_value(self):
        assert TARGET_COL == "pitch_speed_mph"


class TestSyntheticFixtures:
    def test_poi_shape(self, synthetic_poi_df):
        assert len(synthetic_poi_df) == 50
        assert TARGET_COL in synthetic_poi_df.columns
        assert "session_pitch" in synthetic_poi_df.columns

    def test_poi_has_bio_cols(self, synthetic_poi_df):
        for col in KINEMATIC_COLS[:5]:
            assert col in synthetic_poi_df.columns

    def test_hp_shape(self, synthetic_hp_df):
        assert len(synthetic_hp_df) == 100
        assert TARGET_COL in synthetic_hp_df.columns
