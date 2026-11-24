from search import Index, load


def idx():
    return Index(load("corpus.txt"))


def test_top_hit_for_docker():
    assert "Docker" in idx().search("containers for applications")[0][2]


def test_top_hit_for_forecasting():
    assert "Time series" in idx().search("predict future values")[0][2]


def test_no_overlap_returns_empty():
    assert idx().search("zzzzqqqq") == []


def test_k_limits_results():
    assert len(idx().search("model", k=2)) <= 2
