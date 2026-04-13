import pandas as pd
import pytest
from src.modeling.purge_embargo import get_train_times, apply_embargo, purge_and_embargo

@pytest.fixture
def t1():
    # 5 trades, length 2 days each
    dates_t0 = pd.date_range("2024-01-01", periods=5, freq="2d")
    dates_t1 = pd.date_range("2024-01-03", periods=5, freq="2d")
    return pd.Series(dates_t1, index=dates_t0)

@pytest.fixture
def test_times():
    # Test period: Trade 3
    dates_t0 = pd.date_range("2024-01-05", periods=1, freq="1d")
    dates_t1 = pd.date_range("2024-01-07", periods=1, freq="1d")
    return pd.Series(dates_t1, index=dates_t0)

def test_get_train_times(t1, test_times):
    # t1:
    # 1: 01-01 -> 01-03
    # 2: 01-03 -> 01-05
    # 3: 01-05 -> 01-07 (Test block)
    # 4: 01-07 -> 01-09
    # 5: 01-09 -> 01-11

    # Test block starts at 01-05, ends at 01-07
    # Trade 1: 01-01 to 01-03 (Keep)
    # Trade 2: 01-03 to 01-05. Ends AT 01-05. (Keep since it doesn't end strictly after test start)
    # Actually wait. cond1: t0 <= test_t0 AND t1 > test_t0.
    # Trade 2: t0(01-03) <= 01-05. t1(01-05) > 01-05 is False. So kept.
    # Trade 3: 01-05 to 01-07. t0(01-05) >= 01-05 AND t0 <= 01-07. True (cond2). Purged.
    # Trade 4: 01-07 to 01-09. t0(01-07) >= 01-05 AND t0 <= 01-07. True (cond2). Purged.
    # Trade 5: 01-09 to 01-11. t0(01-09) >= 01-05 AND t0 <= 01-07. False. Kept.

    purged = get_train_times(t1, test_times)
    assert len(purged) == 3
    assert t1.index[0] in purged.index # Trade 1
    assert t1.index[1] in purged.index # Trade 2
    assert t1.index[4] in purged.index # Trade 5
    assert t1.index[2] not in purged.index # Trade 3
    assert t1.index[3] not in purged.index # Trade 4

def test_apply_embargo(t1, test_times):
    # Test block ends at 01-07.
    # Let's say we pass step=1 day.
    # Embargo ends at 01-08.
    # Any train_time starting (t0) between 01-07 (exclusive) and 01-08 (inclusive) should be purged.
    # From the original t1:
    # Trade 4 starts at 01-07. (01-07 > 01-07) is False. Kept.
    # Let's make an artificial t1 that starts exactly at 01-08.

    t1_custom = t1.copy()
    t1_custom[pd.Timestamp("2024-01-08")] = pd.Timestamp("2024-01-10")

    embargoed = apply_embargo(t1_custom, test_times, step=pd.Timedelta(days=1))

    assert pd.Timestamp("2024-01-08") not in embargoed.index
    assert pd.Timestamp("2024-01-07") in embargoed.index # t0 > end_test is False

def test_apply_embargo_pct(t1, test_times):
    # If step is None, step = total_time * pct.
    # t1 total time: 01-01 to 01-09 (8 days).
    # If pct = 0.5, step = 4 days.
    # test ends at 01-07. Embargo ends at 01-11.
    # Any train_t0 > 01-07 and <= 01-11 is embargoed.
    # Trade 5 starts at 01-09. Should be embargoed.

    embargoed = apply_embargo(t1, test_times, pct_embargo=0.5)
    assert t1.index[4] not in embargoed.index # Trade 5 starts at 01-09

def test_apply_embargo_zero_pct(t1, test_times):
    embargoed = apply_embargo(t1, test_times, pct_embargo=0.0)
    assert len(embargoed) == len(t1)

def test_purge_and_embargo(t1, test_times):
    # Trade 3, 4 purged by get_train_times.
    # Remaining: 1, 2, 5.
    # Trade 5 starts at 01-09.
    # With pct=0.5, step=4d. Embargo ends at 01-11.
    # Trade 5 purged by embargo.
    # Only 1 and 2 should remain.
    final = purge_and_embargo(t1, test_times, pct_embargo=0.5)
    assert len(final) == 2
    assert t1.index[0] in final.index
    assert t1.index[1] in final.index
