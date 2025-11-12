from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass
class FilterSelection:
    mode: str
    percentile: Optional[float]
    rank: Optional[int]


def render_percentile_rank_filters(total_entries: int) -> FilterSelection:
    options = ["Percentile", "Rank", "None"]
    default_mode = st.session_state.get("filter_mode", "Percentile")
    st.subheader("Percentile / Rank Filter")
    mode = st.radio(
        "Filter mode",
        options,
        horizontal=True,
        index=options.index(default_mode) if default_mode in options else 0,
        key="filter_mode",
    )
    percentile: Optional[float] = None
    rank: Optional[int] = None
    if mode == "Percentile":
        chip_values = [50.0, 25.0, 10.0, 1.0]
        chip_cols = st.columns(len(chip_values))
        for col, pct in zip(chip_cols, chip_values):
            if col.button(f"Top {pct:g}%", key=f"percentile_chip_{pct}"):
                st.session_state["percentile_slider"] = pct
        percentile_default = float(st.session_state.get("percentile_slider", 100.0))
        percentile = st.slider(
            "Show entries through percentile",
            min_value=0.0,
            max_value=100.0,
            value=percentile_default,
            step=0.1,
            key="percentile_slider",
        )
    elif mode == "Rank":
        max_rank = max(int(total_entries), 1)
        default_rank = int(st.session_state.get("rank_slider", max_rank))
        default_rank = max(1, min(default_rank, max_rank))
        rank = st.slider(
            "Max rank",
            min_value=1,
            max_value=max_rank,
            value=default_rank,
            key="rank_slider",
        )
    return FilterSelection(
        mode=mode,
        percentile=percentile if mode == "Percentile" else None,
        rank=rank if mode == "Rank" else None,
    )
