import math
import numpy as np
import pandas as pd
import pytest

from soc_dashboard.analysis import metrics, outcomes, threshold_curve, error_groups
from soc_dashboard.data import filled, source_label


def test_metrics_at_boundary_and_empty_denominators():
    frame = pd.DataFrame({"y_true": [1, 1, 0, 0], "proba": [.2, .1, .2, .1]})
    m = metrics(frame, .2)
    assert [m[k] for k in ["tp", "fp", "fn", "tn"]] == [1, 1, 1, 1]
    assert outcomes(frame, .2).tolist() == ["TP", "FN", "FP", "TN"]
    empty = metrics(frame.iloc[:0], .2)
    assert empty["n"] == 0 and math.isnan(empty["miss"])
    assert math.isnan(metrics(frame, 1)["precision"])
    with pytest.raises(ValueError):
        metrics(pd.DataFrame({"y_true": [None], "proba": [.1]}), .2)


def test_all_threshold_decision_changes_and_monotonicity():
    frame = pd.DataFrame({"y_true": [1, 0, 1, 0], "proba": [0., .2, .9, 1.]})
    curve = threshold_curve(frame)
    assert (curve.fn.diff().dropna() >= 0).all()
    assert (curve.workload.diff().dropna() <= 0).all()
    assert curve.iloc[0].workload == 1 and curve.iloc[-1].workload == 0
    after_boundary = curve[curve.tau.eq(np.nextafter(.2, np.inf))].iloc[0]
    assert after_boundary.fp == 1


def test_missing_enrichment_and_one_class_groups():
    assert filled(pd.Series([None, "", "[]", " null ", '["signal"]'])).tolist() == [False]*4 + [True]
    assert source_label('["MaxPatrol SIEM", "MaxPatrol SIEM"]') == "MaxPatrol SIEM"
    assert source_label('["MaxPatrol SIEM", "Ankey SIEM NG"]') == "Ankey SIEM NG + MaxPatrol SIEM"
    frame = pd.DataFrame({"y_true": [0, 0], "proba": [.1, .9], "rule": ["a", "a"]})
    groups = error_groups(frame, .5, "rule")
    assert math.isnan(groups.iloc[0]["Доля пропусков"])


def test_fast_threshold_curve_matches_direct_counts():
    rng=np.random.default_rng(12)
    df=pd.DataFrame({'proba':rng.choice([0.,.02,.2,.9,1.],size=300),'y_true':rng.integers(0,2,300)})
    for _,row in threshold_curve(df).iterrows():
        direct=metrics(df,row.tau)
        assert all(row[k]==direct[k] for k in ['tp','fp','fn','tn'])
