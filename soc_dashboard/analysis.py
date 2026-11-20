"""Расчёты, независимые от Taipy; пустые знаменатели возвращают NaN."""
import numpy as np
import pandas as pd


def divide(a, b):
    return a / b if b else float("nan")


def metrics(df, tau):
    if not df.y_true.isin([0, 1]).all() or not df.proba.between(0, 1).all():
        raise ValueError("Метрики требуют размеченных инцидентов и вероятностей [0, 1]")
    y, pred = df.y_true.eq(1), df.proba.ge(tau)
    tp, fn = int((y & pred).sum()), int((y & ~pred).sum())
    fp, tn = int((~y & pred).sum()), int((~y & ~pred).sum())
    return dict(n=len(df), tp=tp, fn=fn, fp=fp, tn=tn,
                recall=divide(tp, tp + fn), miss=divide(fn, tp + fn),
                precision=divide(tp, tp + fp), fpr=divide(fp, fp + tn),
                accuracy=divide(tp + tn, len(df)), workload=divide(tp + fp, len(df)))


def outcomes(df, tau):
    y, pred = df.y_true.eq(1), df.proba.ge(tau)
    return pd.Series(np.select([y & pred, y & ~pred, ~y & pred],
                              ["TP", "FN", "FP"], default="TN"), index=df.index)


def threshold_curve(df):
    # Один проход по отсортированным оценкам вместо O(n²) при полном тесте.
    p = df.proba.to_numpy(dtype=float)
    y = df.y_true.to_numpy(dtype=int)
    order = np.argsort(p, kind="stable")
    p, y = p[order], y[order]
    candidates = np.unique(np.r_[0., p, np.nextafter(p, np.inf), np.nextafter(1., 2.)])
    removed = np.searchsorted(p, candidates, side="left")
    cumulative = np.r_[0, np.cumsum(y)]
    fn = cumulative[removed]
    tp = int(y.sum()) - fn
    tn = removed - fn
    fp = len(p) - removed - tp
    def ratio(a,b):
        return np.divide(a,b,out=np.full(len(candidates),np.nan),where=b!=0)
    return pd.DataFrame(dict(tau=candidates,n=len(p),tp=tp,fn=fn,fp=fp,tn=tn,
        recall=ratio(tp,tp+fn),miss=ratio(fn,tp+fn),precision=ratio(tp,tp+fp),
        fpr=ratio(fp,fp+tn),accuracy=ratio(tp+tn,np.full(len(candidates),len(p))),
        workload=ratio(tp+fp,np.full(len(candidates),len(p)))))


def error_groups(df, tau, column):
    rows = []
    for name, group in df.groupby(column, dropna=False):
        m = metrics(group, tau)
        rows.append({"Группа": str(name), "Инциденты": m["n"],
                     "Реальные инциденты": m["tp"] + m["fn"],
                     "Пропущено": m["fn"], "Доля пропусков": m["miss"],
                     "Ложные тревоги": m["fp"]})
    return pd.DataFrame(rows, columns=["Группа", "Инциденты", "Реальные инциденты",
                                      "Пропущено", "Доля пропусков", "Ложные тревоги"])


def pct(value):
    return "нет данных" if pd.isna(value) else f"{value:.1%}"
