"""Собственный бустинг: только поля, доступные при создании инцидента."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'artifacts/own_model'
CATEGORICAL = ['Наименование', 'Критичность', 'Источник инцидента', 'Пиктограмма IRP',
               'Вендор из сием', 'object']
ARRAYS = ['src', 'dst', 'suser', 'duser', 'Команда', 'Алерт', 'processName']
TIME_FIELDS = ['Создан', 'Время первого события', 'Время последнего события', 'Дата создания в источнике']
FEATURE_COLUMNS = list(dict.fromkeys(CATEGORICAL + ARRAYS + TIME_FIELDS + ['CMDLINE']))


def items(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        v = json.loads(str(value))
        return v if isinstance(v, list) else [str(v)]
    except (ValueError, TypeError):
        return [str(value)] if str(value).strip() else []


def features(raw):
    """Явный белый список. Вердикты, решения, статусы и SLA не читаются."""
    result = pd.DataFrame(index=raw.index)
    for c in CATEGORICAL:
        result[c] = raw[c].map(lambda v: ' + '.join(sorted(set(map(str, items(v))))) or 'Нет данных')
    created = pd.to_datetime(raw['Создан'], utc=True, errors='coerce')
    first = pd.to_datetime(raw['Время первого события'], utc=True, errors='coerce')
    last = pd.to_datetime(raw['Время последнего события'], utc=True, errors='coerce')
    source = pd.to_datetime(raw['Дата создания в источнике'], utc=True, errors='coerce')
    result['Час создания'] = created.dt.hour.fillna(-1)
    result['День недели'] = created.dt.dayofweek.fillna(-1)
    # Невозможные времена не маскируются под реальные отрицательные длительности.
    for name, seconds in [('Длительность событий, ч', (last-first).dt.total_seconds()),
                          ('Задержка регистрации, ч', (created-source).dt.total_seconds())]:
        result[name] = (seconds/3600).where(seconds.between(0, 86400*365)).fillna(-1).clip(-1, 24*30)
    for c in ARRAYS:
        result['Число элементов: '+c] = raw[c].map(lambda v: min(len(items(v)), 1000))
    commands = (raw['Команда'].fillna('')+' '+raw['CMDLINE'].fillna('')).str.lower()
    for token in ['powershell', 'nmap', 'zenmap', '-enc', 'http', 'certutil', 'wget']:
        result['Команда содержит '+token] = commands.str.contains(token, regex=False).astype(int)
    result['Длина команд'] = commands.str.len().clip(upper=20000)
    return result


@lru_cache(maxsize=1)
def metadata():
    return json.loads((MODEL_DIR/'metadata.json').read_text())


@lru_cache(maxsize=1)
def load_model():
    import joblib
    return joblib.load(MODEL_DIR/'model.joblib')


def predict(raw):
    bundle = load_model()
    return bundle['model'].predict_proba(bundle['preprocessor'].transform(features(raw)))[:, 1]


@lru_cache(maxsize=1)
def predictions():
    result = pd.read_parquet(MODEL_DIR/'predictions.parquet')
    result['created'] = pd.to_datetime(result.created, utc=True)
    return result
