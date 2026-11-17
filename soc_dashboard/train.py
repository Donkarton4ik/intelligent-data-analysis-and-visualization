"""Обучение: python -m soc_dashboard.train. Сервер Taipy не запускается."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from . import ml
from .data import DATASET, ROOT
from .analysis import metrics, outcomes


def score(frame, tau):
    return {**metrics(frame, tau), 'roc_auc': float(roc_auc_score(frame.y_true, frame.proba)),
            'average_precision': float(average_precision_score(frame.y_true, frame.proba))}


def main():
    practice_path = ROOT / 'data/cards_practice.csv'
    if not practice_path.is_file():
        raise FileNotFoundError(f"Для переобучения нужен {practice_path}. См. data/README.md.")
    practice_ids = pd.read_csv(practice_path, sep=';', usecols=['EntityId'])['EntityId']
    path = DATASET
    raw = pd.read_parquet(path, columns=ml.FEATURE_COLUMNS+['EntityId','Вердикт'])
    raw['created'] = pd.to_datetime(raw['Создан'], utc=True, errors='coerce')
    raw['y_true'] = raw['Вердикт'].map({'False Positive':0,'Решён':1})
    raw = raw[raw.y_true.notna() & raw.created.ge('2025-06-01') & raw.created.lt('2025-10-01')].copy()
    raw.y_true = raw.y_true.astype(int)
    # Все практические инциденты, в том числе team=21, исключаем даже при иной дате.
    train = raw[raw.created.lt('2025-07-01') & ~raw.EntityId.isin(practice_ids)]
    val = raw[raw.created.ge('2025-07-01') & raw.created.lt('2025-08-01') & ~raw.EntityId.isin(practice_ids)]
    test = raw[raw.created.ge('2025-08-01') & raw.created.lt('2025-09-01')]
    fresh = raw[raw.created.ge('2025-09-01')]
    for split in [train,val,test,fresh]:
        if split.y_true.nunique()!=2:
            raise ValueError('Каждый период должен содержать оба класса')
    X = ml.features(train)
    numerical = [c for c in X if c not in ml.CATEGORICAL]
    preprocessor = ColumnTransformer([
        ('category', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10,
                                  max_categories=60, sparse_output=False), ml.CATEGORICAL),
        ('numeric','passthrough',numerical)], verbose_feature_names_out=True)
    matrix = preprocessor.fit_transform(X).astype('float32')
    model = GradientBoostingClassifier(n_estimators=180, learning_rate=.06, max_depth=3,
        min_samples_leaf=15, subsample=.85, random_state=42)
    print(f'Обучение: {len(train)} строк, {matrix.shape[1]} признаков', flush=True)
    model.fit(matrix, train.y_true)
    def forecast(frame):
        p = model.predict_proba(preprocessor.transform(ml.features(frame)).astype('float32'))[:,1]
        return pd.DataFrame({'EntityId':frame.EntityId.values, 'created':frame.created.reset_index(drop=True),
                             'y_true':frame.y_true.values, 'proba':p})
    validation = forecast(val)
    precision, recall, limits = precision_recall_curve(validation.y_true, validation.proba)
    f2 = 5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1], 1e-15)
    tau = float(limits[int(np.argmax(f2))])
    output = ml.MODEL_DIR
    output.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'preprocessor':preprocessor},output/'model.joblib')
    test_pred, fresh_pred = forecast(test), forecast(fresh)
    for frame, name in [(test_pred,'test'), (fresh_pred,'fresh')]:
        frame['split'] = name
        frame['outcome'] = outcomes(frame, tau)
    pd.concat([test_pred,fresh_pred],ignore_index=True).to_parquet(output/'predictions.parquet',index=False)
    with path.open('rb') as dataset:
        source_hash = hashlib.file_digest(dataset, 'sha256').hexdigest()
    information = dict(name='GradientBoosting SOC', seed=42, threshold=tau,
        input_columns=ml.FEATURE_COLUMNS, transformed_features=matrix.shape[1],
        split_ids={n: list(map(int,f.EntityId)) for n,f in [('train',train),('validation',val),('test',test)]},
        periods={'train':'2025-06','validation':'2025-07','test':'2025-08','fresh':'2025-09'},
        validation=score(validation,tau), test=score(test_pred,tau), fresh=score(fresh_pred,tau),
        test_at_05=score(test_pred,.5), baseline_all_alert=metrics(test_pred,0),
        source_sha256=source_hash,
        threshold_policy='max F2 on July validation only; no refit on validation/test',
        caveats=['Разметка только Решён / False Positive', 'Оценки не калиброваны',
                 'Временной разрез не исключает повторные кампании между месяцами'])
    (output/'metadata.json').write_text(json.dumps(information,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:information[k] for k in ['threshold','validation','test','fresh']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
