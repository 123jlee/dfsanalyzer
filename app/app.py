from __future__ import annotations

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure the core package is on the path when running via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from components.controls import FilterSelection, render_percentile_rank_filters
from core import aggregate
from core.ingest import IngestResult, ingest_contest

st.set_page_config(page_title="DFS Contest Analyzer", layout="wide")


def _save_upload(upload: Any) -> Path:
    tmp = NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(upload.getvalue())
    tmp.close()
    return Path(tmp.name)


def _format_lineup(players: Sequence[str]) -> str:
    return " | ".join(players)


def _combo_display(frame: pd.DataFrame, entry_ids: Iterable[int], top_n: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    id_set = set(entry_ids)
    display = frame.copy()
    if id_set:
        display["count_in_current_percentile"] = display["entry_ids"].apply(
            lambda ids: int(sum(1 for value in (ids or []) if value in id_set))
        )
        display = display[display["count_in_current_percentile"] > 0]
    else:
        display["count_in_current_percentile"] = display["frequency"]
    display = display.sort_values(["count_in_current_percentile", "frequency", "best_rank"], ascending=[False, False, True])
    return display.head(top_n)


def _downloadable_csv(frame: pd.DataFrame) -> bytes:
    if frame.empty:
        return b""
    export = frame.copy()
    if "entry_ids" in export.columns:
        export["entry_ids"] = export["entry_ids"].apply(lambda values: ",".join(str(v) for v in values))
    if "players" in export.columns:
        export["players"] = export["players"].apply(lambda values: " | ".join(values))
    return export.to_csv(index=False).encode("utf-8")


def _ensure_session_state() -> None:
    for key in [
        "contest_tables",
        "contest_output_dir",
        "unmatched_players",
        "ingest_time",
    ]:
        st.session_state.setdefault(key, None)


def _store_result(result: IngestResult) -> None:
    st.session_state["contest_tables"] = {name: df.copy() for name, df in result.tables.items()}
    st.session_state["contest_output_dir"] = str(result.output_dir)
    st.session_state["unmatched_players"] = result.unmatched_players
    st.session_state["ingest_time"] = result.ingest_time


def _load_sample() -> None:
    sample_dir = ROOT / "data"
    standings = sample_dir / "sample_standings.csv"
    salaries = sample_dir / "sample_salaries.csv"
    with st.spinner("Ingesting sample contest..."):
        result = ingest_contest(standings, salaries)
    _store_result(result)
    st.success("Sample contest loaded", icon="✅")


def _ingest_uploaded(standings_file, salaries_file) -> None:
    if not standings_file or not salaries_file:
        st.error("Please provide both the contest standings and salary CSV files.")
        return
    standings_path = _save_upload(standings_file)
    salaries_path = _save_upload(salaries_file)
    try:
        with st.spinner("Ingesting contest..."):
            result = ingest_contest(standings_path, salaries_path)
        _store_result(result)
        st.success("Contest ingested successfully", icon="✅")
    finally:
        standings_path.unlink(missing_ok=True)
        salaries_path.unlink(missing_ok=True)


def _render_metrics(entries: pd.DataFrame, contest_meta: pd.DataFrame) -> None:
    total_entries = len(entries)
    unique_users = entries["username"].nunique()
    duplicate_lineups = int((entries["dupe_count"] > 1).sum())
    ingest_time = st.session_state.get("ingest_time")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entries", f"{total_entries:,}")
    col2.metric("Users", f"{unique_users:,}")
    col3.metric("Duplicate Lineups", f"{duplicate_lineups:,}")
    if ingest_time:
        col4.metric("Ingested", ingest_time)
    elif not contest_meta.empty and "ingest_time" in contest_meta.columns:
        col4.metric("Ingested", contest_meta.iloc[0]["ingest_time"])


def _user_lineups(entries: pd.DataFrame, username: str) -> pd.DataFrame:
    subset = entries.loc[entries["username"] == username].copy()
    if subset.empty:
        return subset
    subset["lineup"] = subset["lineup_players"].apply(_format_lineup)
    return subset[[
        "entry_id",
        "rank",
        "percentile",
        "points",
        "dupe_count",
        "salary_sum",
        "lineup",
    ]].sort_values("rank")


def _user_combos(
    username: str,
    combos: Dict[int, pd.DataFrame],
    entry_user_map: Dict[int, str],
    entry_ids: Iterable[int],
    top_n: int,
) -> pd.DataFrame:
    user_ids = {eid for eid in entry_ids if entry_user_map.get(eid) == username}
    if not user_ids:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for frame in combos.values():
        if frame.empty:
            continue
        filtered = frame.copy()
        filtered["count_in_current_percentile"] = filtered["entry_ids"].apply(
            lambda ids: int(sum(1 for value in (ids or []) if value in user_ids))
        )
        filtered = filtered[filtered["count_in_current_percentile"] > 0]
        frames.append(filtered)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["size", "count_in_current_percentile", "best_rank"], ascending=[True, False, True])
    return result.head(top_n)


def main() -> None:
    _ensure_session_state()
    st.title("DFS Contest Analyzer")
    st.caption("Upload DraftKings contest standings and salaries to explore exposures, combos, and stacks.")

    with st.sidebar:
        st.header("Inputs")
        if st.button("Load sample data", use_container_width=True):
            _load_sample()
        with st.form("upload_form"):
            standings_file = st.file_uploader("Contest standings CSV", type="csv", key="standings_upload")
            salaries_file = st.file_uploader("DraftKings salaries CSV", type="csv", key="salaries_upload")
            submitted = st.form_submit_button("Analyze contest", use_container_width=True)
        if submitted:
            _ingest_uploaded(standings_file, salaries_file)
        output_dir = st.session_state.get("contest_output_dir")
        if output_dir:
            st.info(f"Latest output stored in `{output_dir}`")

    tables: Dict[str, pd.DataFrame] | None = st.session_state.get("contest_tables")
    if not tables:
        st.info("Load the sample data or upload a contest to begin.")
        return

    entries = tables["Entries"].copy()
    entries_exploded = tables["EntriesExploded"].copy()
    field_players = tables.get("FieldPlayers", pd.DataFrame()).copy()
    contest_meta = tables.get("ContestMeta", pd.DataFrame()).copy()
    unmatched_players = st.session_state.get("unmatched_players") or []

    combos = {
        size: tables.get(f"Combos{size}", pd.DataFrame()).copy()
        for size in (2, 3, 4)
    }
    team_stacks = tables.get("TeamStacks", pd.DataFrame()).copy()
    game_stacks = tables.get("GameStacks", pd.DataFrame()).copy()

    filter_selection: FilterSelection = render_percentile_rank_filters(len(entries))
    filtered_entries = aggregate.apply_percentile_filter(entries, filter_selection.percentile, filter_selection.rank)
    if filtered_entries.empty:
        st.warning("No entries match the current filter selection.")
    filtered_entry_ids = [int(eid) for eid in filtered_entries["entry_id"].dropna().astype(int).tolist()]
    filtered_exploded = entries_exploded[entries_exploded["entry_id"].isin(filtered_entry_ids)] if filtered_entry_ids else entries_exploded.head(0)
    filtered_user_exposure = aggregate.compute_user_exposure(filtered_entries, filtered_exploded, field_players) if not filtered_entries.empty else pd.DataFrame()

    entry_user_map = {int(row.entry_id): row.username for row in entries.itertuples(index=False) if not pd.isna(row.entry_id)}

    tabs = st.tabs(["Overview", "Users", "Combos", "Field"])

    with tabs[0]:
        _render_metrics(entries, contest_meta)
        histogram_source = filtered_entries if not filtered_entries.empty else entries
        st.plotly_chart(
            px.histogram(histogram_source, x="points", nbins=40, title="Points Distribution"),
            use_container_width=True,
        )
        dupes = filtered_entries.loc[filtered_entries["dupe_count"] > 1].copy()
        if not dupes.empty:
            dupes["lineup"] = dupes["lineup_players"].apply(_format_lineup)
            st.subheader("Duplicate Lineups")
            st.dataframe(
                dupes[["entry_id", "rank", "percentile", "username", "dupe_count", "lineup"]],
                use_container_width=True,
            )
        if unmatched_players:
            with st.expander("Unmatched players", expanded=False):
                st.table(pd.DataFrame(unmatched_players, columns=["Player"]))

    with tabs[1]:
        st.subheader("User Explorer")
        user_search = st.text_input("Search users", value="", placeholder="Start typing a username")
        user_options = sorted(entries["username"].dropna().unique().tolist())
        if not user_options:
            st.info("No users available in this contest.")
        else:
            filtered_users = [user for user in user_options if user_search.lower() in user.lower()]
            options = filtered_users or user_options
            selected_user = st.selectbox("Select user", options, index=0)
            top_n_user = st.slider("Top combos to show", min_value=10, max_value=500, value=100, step=10, key="user_combo_cap")
            if selected_user:
                user_lineups = _user_lineups(filtered_entries, selected_user)
                st.markdown(f"**Lineups for {selected_user} ({len(user_lineups)} shown)**")
                st.dataframe(user_lineups, use_container_width=True)
                user_exposure = filtered_user_exposure.loc[filtered_user_exposure["username"] == selected_user]
                if not user_exposure.empty:
                    st.markdown("**Exposure vs Field**")
                    st.dataframe(
                        user_exposure[[
                            "player",
                            "entry_count",
                            "user_total_lineups",
                            "user_exposure_pct",
                            "field_pct",
                            "delta_vs_field",
                            "best_rank",
                            "max_points",
                        ]].sort_values("user_exposure_pct", ascending=False),
                        use_container_width=True,
                    )
                user_combos = _user_combos(selected_user, combos, entry_user_map, filtered_entry_ids, top_n_user)
                if not user_combos.empty:
                    st.markdown("**User Combos in View**")
                    st.dataframe(user_combos.drop(columns=["entry_ids"], errors="ignore"), use_container_width=True)

    with tabs[2]:
        st.subheader("Combos & Stacks")
        combo_type = st.selectbox("View", ["Name Combos", "Team Stacks", "Game Stacks"], key="combo_type")
        top_n = st.slider("Top rows", min_value=50, max_value=5000, value=500, step=50, key="combo_cap")
        if combo_type == "Name Combos":
            size = st.selectbox("Combo size", [2, 3, 4], index=0, key="combo_size")
            frame = combos.get(size, pd.DataFrame())
            display = _combo_display(frame, filtered_entry_ids, top_n)
        elif combo_type == "Team Stacks":
            size = st.selectbox("Stack size", [2, 3, 4], index=0, key="team_stack_size")
            frame = team_stacks.loc[team_stacks["size"] == size] if not team_stacks.empty else pd.DataFrame()
            display = _combo_display(frame, filtered_entry_ids, top_n)
        else:
            size = st.selectbox("Stack size", [2, 3, 4, 5, 6, 7], index=0, key="game_stack_size")
            frame = game_stacks.loc[game_stacks["size"] == size] if not game_stacks.empty else pd.DataFrame()
            display = _combo_display(frame, filtered_entry_ids, top_n)
        if display.empty:
            st.info("No combos available for the current selection.")
        else:
            st.dataframe(display.drop(columns=["entry_ids"], errors="ignore"), use_container_width=True)
            st.download_button(
                "Download CSV",
                data=_downloadable_csv(display),
                file_name=f"{combo_type.replace(' ', '_').lower()}_{size}.csv",
                mime="text/csv",
            )

    with tabs[3]:
        st.subheader("Field Ownership")
        if field_players.empty:
            st.info("Field ownership table not available in this contest export.")
        else:
            player_search = st.text_input("Search players", value="", key="field_search", placeholder="Player name")
            field_view = field_players.copy()
            if player_search:
                mask = field_view["player"].str.contains(player_search, case=False, na=False)
                field_view = field_view[mask]
            field_view = field_view.sort_values("field_pct", ascending=False)
            st.dataframe(field_view, use_container_width=True)
            st.download_button(
                "Download Field CSV",
                data=_downloadable_csv(field_view),
                file_name="field_players.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
