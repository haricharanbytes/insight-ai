import io

import pandas as pd

from src.agent.demo_router import run_demo
from src.data.profiler import profile_dataset

DF = pd.read_csv(io.StringIO(
    "Region,Sales,Profit,Order Date\n"
    "East,100,10,2020-01-15\nEast,200,-20,2020-01-20\nWest,50,5,2020-02-10\n"
    "West,150,15,2020-03-05\nNorth,400,60,2020-03-30\nNorth,300,40,2020-04-02\n"
    "East,120,12,2020-04-10\nWest,80,8,2020-05-01\nNorth,250,30,2020-05-20\nEast,90,9,2020-06-11\n"))
PROFILE = profile_dataset(DF)


def ask(q):
    return run_demo(DF, PROFILE, q)


def test_ranking_uses_real_numbers():
    r = ask("Which Region has the highest Sales?")
    assert r["mode"] == "demo" and r["steps"][0]["tool"] == "group_by_aggregate"
    assert "North is first" in r["answer"] and "(950)" in r["answer"]


def test_lowest_sorts_ascending():
    r = ask("Which Region has the lowest total Sales?")
    assert "West is first" in r["answer"]


def test_trend_question():
    r = ask("Show monthly Sales trend")
    assert r["steps"][0]["tool"] == "analyze_time_series"
    assert r["steps"][0]["params"]["freq"] == "M"


def test_correlation_and_outliers_and_quality():
    assert ask("correlation between Sales and Profit")["steps"][0]["tool"] == "calculate_correlation"
    assert ask("any outliers in Profit?")["steps"][0]["params"] == {"column": "Profit"}
    q = ask("how is the data quality?")
    assert q["steps"] == [] and "10 rows" in q["answer"]


def test_unsupported_is_honest_not_invented():
    r2 = ask("total Salary by Region")
    assert not any(s["params"].get("metrics", [{}])[0].get("column") == "Salary" for s in r2["steps"])