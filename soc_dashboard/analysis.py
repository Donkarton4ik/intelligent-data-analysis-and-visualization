"""Форматирование долей и обработка пустых знаменателей."""
import pandas as pd


def divide(a, b):
    return a / b if b else float("nan")


def pct(value):
    return "нет данных" if pd.isna(value) else f"{value:.1%}"
