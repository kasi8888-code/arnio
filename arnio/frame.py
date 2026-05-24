"""
arnio.frame
ArFrame — the core data container wrapping the C++ Frame.
"""

from __future__ import annotations

from ._core import _Frame


class ArFrame:
    """Lightweight columnar data container backed by C++."""

    __slots__ = ("_frame",)

    def __init__(self, cpp_frame: _Frame) -> None:
        self._frame = cpp_frame

    # --- Properties ---

    @property
    def shape(self) -> tuple[int, int]:
        """Row and column count.

        Returns
        -------
        tuple[int, int]
            (number_of_rows, number_of_columns)
        """
        return self._frame.shape()

    @property
    def columns(self) -> list[str]:
        """Column names.

        Returns
        -------
        list[str]
            List of column names in order.
        """
        return self._frame.column_names()

    @property
    def dtypes(self) -> dict[str, str]:
        """Column name → inferred type.

        Returns
        -------
        dict[str, str]
            Mapping of column names to their data types.
        """
        return self._frame.dtypes()

    # --- Methods ---

    def memory_usage(self, deep: bool = False) -> int:
        """Total bytes consumed in memory.

        Parameters
        ----------
        deep : bool, optional
            When ``False`` (default), returns the same value as the original
            ``memory_usage()`` API: for string columns, counts each string's
            *reserved* capacity (``s.capacity()``). This is fully
            backward-compatible with existing callers.

            When ``True``, performs a deeper inspection of string columns:
            counts each string's *actual used* bytes (``s.size()``) rather
            than the reserved capacity. This gives a tighter, more accurate
            estimate of real memory consumed and will typically return a
            smaller number than the default path for string-heavy frames
            because unused reserved capacity is excluded.

            For non-string columns (int64, float64, bool) the result is
            identical in both modes because those types store all data inline.

        Returns
        -------
        int
            Total memory usage in bytes.

        Examples
        --------
        >>> frame = ar.read_csv("data.csv")
        >>> frame.memory_usage()                 # backward-compatible default
        2048
        >>> frame.memory_usage(deep=True)        # tighter actual-bytes estimate
        1800
        """
        return self._frame.memory_usage(deep)

    # --- Dunder methods ---

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._frame.num_rows()

    def __repr__(self) -> str:
        """Return a string representation of the ArFrame."""
        rows, cols = self.shape
        return f"ArFrame({rows} rows × {cols} cols)"

    def __str__(self) -> str:
        """Return a detailed string summary of the ArFrame."""
        lines = [f"ArFrame: {self.shape[0]} rows × {self.shape[1]} columns"]
        lines.append(f"Columns: {self.columns}")
        lines.append(f"DTypes:  {self.dtypes}")
        lines.append(f"Memory:  {self.memory_usage()} bytes")
        return "\n".join(lines)
