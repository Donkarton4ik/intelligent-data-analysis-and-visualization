"""Загрузка и фильтрация учебной выгрузки SOC."""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/iavd_ml_dataset.parquet"
ALL = "Все"
LABELS = {"Решён": 1, "False Positive": 0}
SEVERITY_LEVELS = ("Критическая", "Высокая", "Средняя", "Низкая")



def filled(series):
    """Пустой JSON-массив не означает наличие обогащения."""
    return ~series.fillna("").astype(str).str.strip().str.lower().isin(
        ["", "[]", "{}", "null", "none", "nan"]
    )


def source_label(value):
    if value is None or pd.isna(value):
        return "Не указано"
    try:
        decoded = json.loads(str(value))
        if isinstance(decoded, list):
            return " + ".join(sorted({str(v) for v in decoded if v})) or "Не указано"
    except (ValueError, TypeError):
        pass
    return str(value).strip() or "Не указано"


@lru_cache(maxsize=1)
def soc_cards():
    columns = ["EntityId", "Создан", "Вердикт", "Источник инцидента", "Наименование",
               "Критичность", "Тип инцидента", "Время решения", "Дата закрытия (факт)",
               "Дата взятия в работу", "Ответственный", "Группа устранения", "Hostname Цели",
               "SLA по взятию в работу Атаки"]
    if not DATASET.is_file():
        raise FileNotFoundError(f"Положите iavd_ml_dataset.parquet в {DATASET.parent}. См. data/README.md.")
    df = pd.read_parquet(DATASET, columns=columns)
    df["created"] = pd.to_datetime(df["Создан"], utc=True, errors="coerce")
    # Сутки во всех разделах определяем одинаково, в UTC, как в исходном логе.
    df["day"] = df.created.dt.floor("D").dt.tz_localize(None)
    df["month"] = df.created.dt.strftime("%Y-%m")
    df["label"] = df["Вердикт"].map(LABELS)
    df["Источник инцидента"] = df["Источник инцидента"].map(source_label)
    df["Тип инцидента"] = df["Тип инцидента"].map(source_label)
    df["verdict_group"] = df["Вердикт"].map(
        {"Решён": "Инцидент", "False Positive": "Ложное срабатывание"}
    ).fillna("Другой исход")
    df.loc[~filled(df["Вердикт"]), "verdict_group"] = "Нет вердикта"
    for col in ["Источник инцидента", "Наименование", "Критичность", "Тип инцидента"]:
        df[col] = df[col].fillna("Не указано")
    hours = pd.to_numeric(df["Время решения"], errors="coerce") / 3600
    closed = pd.to_datetime(df["Дата закрытия (факт)"], utc=True, errors="coerce")
    df["state"] = np.where(closed.notna(), "Закрыт", "Открыт")
    # Физическая граница: длительность не может превышать всё окно выгрузки.
    end = max(df.created.max(), closed.max())
    max_hours = (end - df.created.min()).total_seconds() / 3600
    df["duration_invalid"] = hours.notna() & ~hours.between(0, max_hours)
    df["duration_h"] = hours.where(hours.between(0, max_hours) & closed.notna() & df.label.notna())
    return df


def select_soc(df, month=ALL, source=ALL, criticality=ALL, verdict=ALL,
               start=None, end=None, status=ALL, incident_type=ALL):
    df = between_dates(df, start, end)
    mask = pd.Series(True, index=df.index)
    if month != ALL:
        mask &= df.month.eq(month)
    if source != ALL:
        mask &= df["Источник инцидента"].eq(source)
    if criticality != ALL:
        mask &= df["Критичность"].eq(criticality)
    if verdict != ALL:
        mask &= df.verdict_group.eq(verdict)
    if status != ALL:
        mask &= df.state.eq(status)
    if incident_type != ALL:
        mask &= df["Тип инцидента"].eq(incident_type)
    return df.loc[mask]


def date_limits(df):
    return df.created.min().to_pydatetime().replace(tzinfo=None), df.created.max().to_pydatetime().replace(tzinfo=None)


def between_dates(df, start=None, end=None):
    mask = pd.Series(True, index=df.index)
    if start is not None:
        lo = pd.Timestamp(pd.Timestamp(start).date(), tz="UTC")
        mask &= df.created.ge(lo)
    if end is not None:
        hi = pd.Timestamp(pd.Timestamp(end).date(), tz="UTC") + pd.Timedelta(days=1)
        mask &= df.created.lt(hi)
    if start is not None and end is not None and pd.Timestamp(start).date() > pd.Timestamp(end).date():
        raise ValueError("Начало периода позже конца")
    return df.loc[mask]
