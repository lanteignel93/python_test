from string import ascii_letters, digits

import hypothesis.strategies as st
import polars as pl
from hypothesis import given
from polars import NUMERIC_DTYPES
from polars.testing.parametric import column, dataframes, lists

id_chars = ascii_letters + digits


@given(
    dataframes(
        cols=5,
        allow_null=True,
        allowed_dtypes=NUMERIC_DTYPES,
    )
)
def test_numeric(df: pl.DataFrame):
    assert all(df[col].dtype.is_numeric() for col in df.columns)

    # Example frame:
    # ┌──────┬────────┬───────┬────────────┬────────────┐
    # │ col0 ┆ col1   ┆ col2  ┆ col3       ┆ col4       │
    # │ ---  ┆ ---    ┆ ---   ┆ ---        ┆ ---        │
    # │ u8   ┆ i16    ┆ u16   ┆ i32        ┆ f64        │
    # ╞══════╪════════╪═══════╪════════════╪════════════╡
    # │ 54   ┆ -29096 ┆ 485   ┆ 2147483647 ┆ -2.8257e14 │
    # │ null ┆ 7508   ┆ 37338 ┆ 7264       ┆ 1.5        │
    # │ 0    ┆ 321    ┆ null  ┆ 16996      ┆ NaN        │
    # │ 121  ┆ -361   ┆ 63204 ┆ 1          ┆ 1.1443e235 │
    # └──────┴────────┴───────┴────────────┴────────────┘


@given(
    dataframes(
        cols=[
            column("id", strategy=st.text(min_size=4, max_size=4, alphabet=id_chars)),
            column("ccy", strategy=st.sampled_from(["GBP", "EUR", "JPY", "USD"])),
            column("price", strategy=st.floats(min_value=0.0, max_value=1000.0)),
        ],
        min_size=5,
        lazy=True,
    )
)
def test_price_calculations(lf: pl.LazyFrame):
    df = lf.collect()
    pl.testing.assert_series_not_equal(
        df.select("id").to_series(), df.select("price").to_series()
    )

    # Example frame:
    # ┌──────┬─────┬─────────┐
    # │ id   ┆ ccy ┆ price   │
    # │ ---  ┆ --- ┆ ---     │
    # │ str  ┆ str ┆ f64     │
    # ╞══════╪═════╪═════════╡
    # │ A101 ┆ GBP ┆ 1.1     │
    # │ 8nIn ┆ JPY ┆ 1.5     │
    # │ QHoO ┆ EUR ┆ 714.544 │
    # │ i0e0 ┆ GBP ┆ 0.0     │
    # │ 0000 ┆ USD ┆ 999.0   │
    # └──────┴─────┴─────────┘


@st.composite
def uint8_pairs(draw: st.DrawFn):
    uints = lists(pl.UInt8, size=2)
    pairs = list(zip(draw(uints), draw(uints)))
    return [sorted(ints) for ints in pairs]


@given(
    dataframes(
        cols=[
            column("colx", strategy=uint8_pairs()),
            column("coly", strategy=uint8_pairs()),
            column("colz", strategy=uint8_pairs()),
        ],
        min_size=3,
        max_size=3,
    )
)
def test_miscellaneous(df: pl.DataFrame):
    pl.testing.assert_frame_not_equal(df.select("colx"), df.select("coly"))

    # Example frame:
    # ┌─────────────────────────┬─────────────────────────┬──────────────────────────┐
    # │ colx                    ┆ coly                    ┆ colz                     │
    # │ ---                     ┆ ---                     ┆ ---                      │
    # │ list[list[i64]]         ┆ list[list[i64]]         ┆ list[list[i64]]          │
    # ╞═════════════════════════╪═════════════════════════╪══════════════════════════╡
    # │ [[143, 235], [75, 101]] ┆ [[143, 235], [75, 101]] ┆ [[31, 41], [57, 250]]    │
    # │ [[87, 186], [174, 179]] ┆ [[87, 186], [174, 179]] ┆ [[112, 213], [149, 221]] │
    # │ [[23, 85], [7, 86]]     ┆ [[23, 85], [7, 86]]     ┆ [[22, 255], [27, 28]]    │
    # └─────────────────────────┴─────────────────────────┴──────────────────────────┘
