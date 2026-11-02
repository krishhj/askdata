import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from askdata import config


def _column_kinds(df: pd.DataFrame) -> dict[str, list[str]]:
    kinds = {"numeric": [], "datetime": [], "text": []}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            kinds["text"].append(col)
        elif pd.api.types.is_numeric_dtype(series):
            kinds["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            kinds["datetime"].append(col)
        else:
            kinds["text"].append(col)
    return kinds


def chart_kind(df: pd.DataFrame | None) -> str | None:
    """Pick "metric", "line", "bar", "scatter" or None from the result's shape."""
    if df is None or df.empty:
        return None
    kinds = _column_kinds(df)
    n_numeric, n_datetime, n_text = (len(kinds[k]) for k in ("numeric", "datetime", "text"))

    if df.shape == (1, 1) and n_numeric == 1:
        return "metric"
    if n_datetime == 1 and n_numeric >= 1 and n_text == 0 and len(df) > 1:
        return "line"
    if n_text == 1 and n_numeric == 1 and n_datetime == 0 and len(df) <= config.BAR_MAX_ROWS:
        return "bar"
    if n_numeric == 2 and df.shape[1] == 2 and len(df) > 1:
        return "scatter"
    return None


def _readable(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _bar(df: pd.DataFrame, label: str, value: str, labels: dict) -> Figure:
    if df[label].astype(str).str.len().max() <= config.LONG_LABEL_CHARS:
        return px.bar(df, x=label, y=value, labels=labels)
    fig = px.bar(df, x=value, y=label, orientation="h", labels=labels)
    # Plotly draws the first row at the bottom; reverse so a ranking reads top-down.
    fig.update_yaxes(autorange="reversed", title=None)
    return fig


def _style(fig: Figure) -> Figure:
    fig.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"title": None, "orientation": "h", "y": 1.08},
        hoverlabel={"namelength": -1},
        separators=".,",
        showlegend=len(fig.data) > 1,
    )
    # Thousands separators, so 1233131.72 reads as 1,233,131.72.
    fig.update_xaxes(separatethousands=True)
    fig.update_yaxes(separatethousands=True)
    return fig


def pick_chart(df: pd.DataFrame | None) -> Figure | None:
    """A Plotly figure suited to the result, or None for a metric or no chart."""
    kind = chart_kind(df)
    if kind in (None, "metric"):
        return None

    kinds = _column_kinds(df)
    labels = {col: _readable(col) for col in df.columns}
    if kind == "line":
        x = kinds["datetime"][0]
        fig = px.line(df.sort_values(x), x=x, y=kinds["numeric"], markers=True, labels=labels)
        if len(kinds["numeric"]) == 1:
            fig.update_yaxes(title=labels[kinds["numeric"][0]])
    elif kind == "bar":
        fig = _bar(df, kinds["text"][0], kinds["numeric"][0], labels)
    else:
        x, y = kinds["numeric"]
        fig = px.scatter(df, x=x, y=y, labels=labels)
    return _style(fig)
