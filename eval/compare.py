import datetime
import math
from collections import Counter
from decimal import Decimal
from itertools import product

import numpy as np
import pandas as pd

from askdata import config


def _normalise(value):
    # Makes equal answers compare equal: rounded numbers, ISO dates, trimmed text.
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, Decimal, np.integer, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        # Adding 0.0 turns -0.0 into 0.0.
        return round(number, 2) + 0.0
    if isinstance(value, (datetime.date, pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return str(value).strip()


def _columns(df: pd.DataFrame) -> list[list]:
    # By position, because generated column names rarely match the gold ones.
    return [[_normalise(v) for v in df.iloc[:, i]] for i in range(df.shape[1])]


def _same(a: list, b: list, ordered: bool) -> bool:
    return a == b if ordered else Counter(a) == Counter(b)


def results_match(gold_df: pd.DataFrame, pred_df: pd.DataFrame, ordered: bool,
                  allow_extra_columns: bool | None = None) -> bool:
    """True if pred_df holds the same rows as gold_df, ignoring column names and order."""
    if allow_extra_columns is None:
        allow_extra_columns = config.EVAL_ALLOW_EXTRA_COLUMNS
    if len(gold_df) != len(pred_df) or pred_df.shape[1] < gold_df.shape[1]:
        return False
    if not allow_extra_columns and pred_df.shape[1] != gold_df.shape[1]:
        return False

    gold_cols, pred_cols = _columns(gold_df), _columns(pred_df)
    # Narrow the search: a gold column can only map to a predicted column with the same values.
    candidates = [
        [j for j, pred in enumerate(pred_cols) if _same(gold, pred, ordered)]
        for gold in gold_cols
    ]
    gold_rows = list(zip(*gold_cols))
    for mapping in product(*candidates):
        if len(set(mapping)) < len(mapping):
            continue
        # Columns can match one by one while rows still pair up differently.
        pred_rows = list(zip(*(pred_cols[j] for j in mapping)))
        if _same(gold_rows, pred_rows, ordered):
            return True
    return False
