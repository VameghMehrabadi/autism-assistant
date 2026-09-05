import numpy as np
import pandas as pd

from facts import CATEGORY_KEYS
from gap_analysis import build_report


def test_gap_report_counts_multi_label_even_when_top1_is_zero():
    n_cats = len(CATEGORY_KEYS)
    d_idx = CATEGORY_KEYS.index("D")
    e_idx = CATEGORY_KEYS.index("E")
    f_idx = CATEGORY_KEYS.index("F")

    scores = np.zeros((3, n_cats), dtype=float)
    scores[0, f_idx] = 0.60
    scores[0, d_idx] = 0.50
    scores[1, f_idx] = 0.55
    scores[1, e_idx] = 0.40
    scores[2, f_idx] = 0.70

    df = pd.DataFrame({
        "top_label": ["F", "F", "F"],
        "top_score": [0.60, 0.55, 0.70],
        "labels": ["F|D", "F|E", "F"],
    })

    report = build_report(df, scores, threshold=0.35)
    assert report["per_category"]["F"]["count"] == 3
    assert report["per_category"]["D"]["count"] == 0
    assert report["per_category"]["E"]["count"] == 0
    assert report["per_category"]["D"]["multi_count"] == 1
    assert report["per_category"]["E"]["multi_count"] == 1
    assert report["per_category"]["D"]["global_mean_score"] > 0
    assert report["per_category"]["D"]["mean_score"] == 0.0
