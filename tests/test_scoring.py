"""Tests for MechanicsScorer."""
from pitchlens.analytics.scoring import MechanicsScorer, MechanicsScores


class TestMechanicsScorer:
    def test_fit_and_score(self, synthetic_poi_df):
        scorer = MechanicsScorer()
        scorer.fit(synthetic_poi_df)

        scores = scorer.score(synthetic_poi_df.iloc[0])

        assert isinstance(scores, MechanicsScores)
        for attr in [
            "arm_action", "block", "posture", "rotation", "momentum", "overall",
        ]:
            val = getattr(scores, attr)
            assert 0 <= val <= 100, f"{attr}={val} out of 0-100 range"

    def test_as_dict(self, synthetic_poi_df):
        scorer = MechanicsScorer().fit(synthetic_poi_df)
        scores = scorer.score(synthetic_poi_df.iloc[0])
        d = scores.as_dict()
        assert "arm_action" in d
        assert "overall" in d

    def test_summary_string(self, synthetic_poi_df):
        scorer = MechanicsScorer().fit(synthetic_poi_df)
        scores = scorer.score(synthetic_poi_df.iloc[0])
        text = scores.summary()
        assert "Overall:" in text
        assert "/100" in text

    def test_top_improvements(self, synthetic_poi_df):
        scorer = MechanicsScorer().fit(synthetic_poi_df)
        improvements = scorer.top_improvements(synthetic_poi_df.iloc[0], n=3)
        assert len(improvements) <= 3
        for cat, var, pct in improvements:
            assert isinstance(cat, str)
            assert isinstance(var, str)
            assert 0 <= pct <= 100

    def test_score_cohort(self, synthetic_poi_df):
        scorer = MechanicsScorer().fit(synthetic_poi_df)
        scored = scorer.score_cohort(synthetic_poi_df)
        assert "overall" in scored.columns
        assert len(scored) == len(synthetic_poi_df)

    def test_injury_flags_format(self, synthetic_poi_df):
        scorer = MechanicsScorer().fit(synthetic_poi_df)
        scores = scorer.score(synthetic_poi_df.iloc[0])
        for key, val in scores.injury_flags.items():
            assert isinstance(key, str)
            assert isinstance(val, str)
