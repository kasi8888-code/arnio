"""Tests for ArFrame.memory_usage(deep=False/True)."""

from __future__ import annotations

import pandas as pd

import arnio as ar


class TestMemoryUsageShallow:
    """memory_usage() with default deep=False — backward-compatible behaviour."""

    def test_returns_positive_int_for_int_frame(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage() > 0

    def test_returns_positive_int_for_float_frame(self):
        df = pd.DataFrame({"x": [1.1, 2.2, 3.3]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage() > 0

    def test_returns_positive_int_for_bool_frame(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage() > 0

    def test_returns_positive_int_for_string_frame(self):
        df = pd.DataFrame({"name": ["alice", "bob", "carol"]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage() > 0

    def test_returns_positive_int_for_mixed_frame(self):
        df = pd.DataFrame({"name": ["alice", "bob"], "age": [30, 25]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage() > 0

    def test_empty_frame_returns_nonnegative(self):
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        assert frame.memory_usage() >= 0

    def test_explicit_false_matches_default(self):
        """memory_usage(deep=False) must equal memory_usage()."""
        df = pd.DataFrame({"text": ["hello", "world"]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=False) == frame.memory_usage()


class TestMemoryUsageDeep:
    """memory_usage(deep=True) — tighter actual-bytes estimate for string columns."""

    def test_deep_less_than_or_equal_shallow_for_string_frame(self):
        """For string frames deep=True <= deep=False.

        deep=True counts s.size() (actual bytes used); deep=False counts
        s.capacity() (reserved bytes). capacity >= size always.
        """
        df = pd.DataFrame({"text": ["alice", "bob", "carol"]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=True) <= frame.memory_usage(deep=False)

    def test_deep_equals_shallow_for_int_column(self):
        """Numeric columns: deep has no extra effect."""
        df = pd.DataFrame({"n": [1, 2, 3]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=True) == frame.memory_usage(deep=False)

    def test_deep_equals_shallow_for_float_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=True) == frame.memory_usage(deep=False)

    def test_deep_equals_shallow_for_bool_column(self):
        df = pd.DataFrame({"flag": [True, False]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=True) == frame.memory_usage(deep=False)

    def test_deep_less_than_or_equal_shallow_for_mixed_frame(self):
        """Mixed string+numeric frame: deep <= shallow."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=True) <= frame.memory_usage(deep=False)

    def test_longer_strings_use_more_deep_memory(self):
        """Frames with longer strings must report higher deep memory."""
        short_df = pd.DataFrame({"text": ["a", "b", "c"]})
        long_df = pd.DataFrame({"text": ["a" * 100, "b" * 100, "c" * 100]})
        short_frame = ar.from_pandas(short_df)
        long_frame = ar.from_pandas(long_df)
        assert long_frame.memory_usage(deep=True) > short_frame.memory_usage(deep=True)

    def test_deep_returns_int(self):
        df = pd.DataFrame({"text": ["hello"]})
        frame = ar.from_pandas(df)
        assert isinstance(frame.memory_usage(deep=True), int)

    def test_empty_frame_deep_returns_nonnegative(self):
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        assert frame.memory_usage(deep=True) >= 0

    def test_null_string_column_deep_does_not_crash(self):
        """deep=True on a column with null strings must not raise."""
        df = pd.DataFrame({"text": [None, "hello", None]})
        frame = ar.from_pandas(df)
        result = frame.memory_usage(deep=True)
        assert result >= 0
