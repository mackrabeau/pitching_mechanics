"""Tests for PeerMatcher."""
from pitchlens.analytics.peer_match import PeerMatcher, PitcherComp


class TestPeerMatcher:
    def test_fit_and_find(self, synthetic_poi_df):
        matcher = PeerMatcher()
        matcher.fit(synthetic_poi_df)

        comps = matcher.find_comps(synthetic_poi_df.iloc[0], n=3)

        assert len(comps) <= 3
        assert all(isinstance(c, PitcherComp) for c in comps)
        assert all(0 <= c.similarity <= 1 for c in comps)

    def test_similarity_ordering(self, synthetic_poi_df):
        matcher = PeerMatcher().fit(synthetic_poi_df)
        comps = matcher.find_comps(synthetic_poi_df.iloc[0], n=5)
        sims = [c.similarity for c in comps]
        assert sims == sorted(sims, reverse=True)

    def test_velo_range(self, synthetic_poi_df):
        matcher = PeerMatcher().fit(synthetic_poi_df)
        vr = matcher.velo_range_for_mechanics(synthetic_poi_df.iloc[0], n=5)
        assert "mean" in vr
        assert "min" in vr
        assert "max" in vr
        assert vr["min"] <= vr["mean"] <= vr["max"]

    def test_level_breakdown(self, synthetic_poi_df):
        matcher = PeerMatcher().fit(synthetic_poi_df)
        breakdown = matcher.level_breakdown(synthetic_poi_df.iloc[0], n=10)
        assert "avg_velo" in breakdown.columns
        assert "count" in breakdown.columns

    def test_pitcher_comp_str(self, synthetic_poi_df):
        matcher = PeerMatcher().fit(synthetic_poi_df)
        comps = matcher.find_comps(synthetic_poi_df.iloc[0], n=1)
        assert "mph" in str(comps[0])
