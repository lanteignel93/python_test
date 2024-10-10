import pandas as pd


def test_frame_equal_ignore_dtypes():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def test_series_equal():
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([1, 2, 3, 4])
    pd.testing.assert_series_equal(a, b)


def test_index_equal():
    a = pd.Index([1, 2, 3])
    b = pd.Index([1, 2, 3])
    pd.testing.assert_index_equal(a, b)
