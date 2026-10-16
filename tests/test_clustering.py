import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.clustering import cluster_data
from src.analysis.common import AnalysisError


def blobs(centers, per=40, spread=0.4, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.normal(c, spread, size=(per, len(c))) for c in centers])
    return pd.DataFrame(pts, columns=["a", "b"][: len(centers[0])])


THREE = blobs([(0, 0), (10, 10), (20, 0)])
TWO = blobs([(0, 0), (12, 12)])


def code_of(df, **kw):
    with pytest.raises(AnalysisError) as exc:
        cluster_data(df, **kw)
    return exc.value.code


def test_auto_k_finds_the_true_number_of_blobs():
    out = cluster_data(THREE)
    assert out["result"]["k"] == 3 and out["result"]["silhouette"] > 0.7
    assert sorted(out["result"]["sizes"]) == [40, 40, 40]
    assert out["rows_used"] == 120


def test_two_blobs_and_given_k():
    assert cluster_data(TWO)["result"]["k"] == 2
    out = cluster_data(THREE, k=2)
    assert out["result"]["k"] == 2 and out["result"]["silhouette_by_k"].keys() == {"2"}
    assert any("k was given" in out["method"] for _ in [0])


def test_profile_means_are_in_original_units():
    out = cluster_data(THREE)
    t = out["table"]
    rows = [dict(zip(t["columns"], r)) for r in t["rows"]]
    mean_a = sorted(round(r["mean_a"]) for r in rows)
    assert mean_a == [0, 10, 20]
    assert sum(r["share_pct"] for r in rows) == pytest.approx(100, abs=0.1)


def test_pca_payload_is_bounded_and_serialisable():
    big = blobs([(0, 0), (10, 10)], per=1500)
    out = cluster_data(big)
    pca = out["result"]["pca"]
    assert len(pca["x"]) == 2000 and pca["sampled"] is True and len(pca["variance_explained_pct"]) == 2
    json.dumps(out, allow_nan=False)


def test_uniform_noise_gets_a_weak_structure_warning():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.uniform(0, 1, size=(300, 2)), columns=["a", "b"])
    out = cluster_data(noise)
    assert out["result"]["silhouette"] < 0.5
    assert out["result"]["note"] and "not necessarily natural groups" in out["result"]["note"]


def test_identifier_and_text_columns_are_ignored():
    d = THREE.assign(**{"Row ID": range(120), "name": ["x"] * 120})
    out = cluster_data(d)
    assert out["result"]["columns"] == ["a", "b"] and any("Row ID" in w for w in out["warnings"])


def test_missing_rows_are_dropped_and_reported():
    d = THREE.copy()
    d.loc[0:4, "a"] = np.nan
    out = cluster_data(d)
    assert out["rows_used"] == 115 and any("5 rows" in w for w in out["warnings"])


def test_reproducible():
    assert cluster_data(THREE)["result"]["sizes"] == cluster_data(THREE)["result"]["sizes"]


def test_refusals():
    assert code_of(THREE.head(20)) == "insufficient_data"
    assert code_of(THREE, k=1) == "invalid_parameter"
    assert code_of(THREE, k=20) == "invalid_parameter"
    assert code_of(THREE, columns=["a"]) == "invalid_parameter"
    assert code_of(THREE, columns=["a", "nope"]) == "column_not_found"
    assert code_of(THREE.assign(t="x"), columns=["a", "t"]) == "invalid_column_type"
    assert code_of(THREE.assign(c=1.0), columns=["a", "c"]) == "no_variation"
    assert code_of(THREE, filters=[{"column": "a", "op": "gt", "value": 1e6}]) == "no_rows"
    assert code_of(THREE.head(40), k=5) == "invalid_parameter"      # clusters would average 8 rows


def test_input_not_modified():
    before = THREE.copy()
    cluster_data(THREE)
    pd.testing.assert_frame_equal(THREE, before)