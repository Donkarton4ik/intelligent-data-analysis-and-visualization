"""Задание 2. Доверие к ранжированию с учётом цены пропуска."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import ALL, SEVERITY_LEVELS, model_cards, select_model
from .ml import metadata
from .analysis import metrics, outcomes, error_groups, pct, threshold_curve
from .charts import style, RED, TEAL, empty
from .ui import intro, kpi, panel

def agreement_chart(m):
    if not m['n']:
        return empty('', 'В выбранном срезе нет размеченных карточек')
    codes = (('FN', 'TP'), ('TN', 'FP'))
    counts = np.array([[m[code.lower()] for code in row] for row in codes])
    totals=counts.sum(axis=1,keepdims=True)
    shares=np.divide(counts,totals,out=np.full(counts.shape,np.nan),where=totals>0)
    text = [
        [f'{count:,}<br>{pct(share)}<br>{code}'.replace(',', ' ') if pd.notna(share) else f'—<br>{code}'
         for count, share, code in zip(row_counts, row_shares, row_codes)]
        for row_counts, row_shares, row_codes in zip(counts, shares, codes)
    ]
    fig=go.Figure(go.Heatmap(z=shares,x=['Ниже порога','Направить<br>на разбор'],
        y=['Реальный<br>инцидент','Ложная<br>тревога'],text=text,texttemplate='%{text}',textfont_size=14,
        colorscale=[[0,'#f5f5f5'],[1,'#bdbdbd']],zmin=0,zmax=1,showscale=False,xgap=3,ygap=3,
        customdata=counts,hovertemplate='Вердикт: %{y}<br>Модель: %{x}<br>Карточек: %{customdata:,d}'
                                      '<br>Доля от вердикта: %{z:.1%}<extra></extra>'))
    style(fig,'','Решение модели','Вердикт аналитика')
    fig.update_yaxes(autorange='reversed',showgrid=False)
    fig.add_shape(type='rect',x0=-.5,x1=.5,y0=-.5,y1=.5,line=dict(color=RED,width=2))
    return fig


def workload_chart(df,tau):
    if df.empty:
        return empty('')
    thresholds=np.unique(np.r_[np.linspace(0,1,201),tau])
    curve=threshold_curve(df)
    values=curve.iloc[np.searchsorted(curve.tau,thresholds,side='left')]
    fig=go.Figure()
    for column,name,color in [('miss','Пропущено реальных',RED),('workload','Направлено на разбор',TEAL)]:
        if values[column].notna().any():
            fig.add_scatter(x=thresholds,y=values[column],mode='lines',name=name,line_color=color,
                            hovertemplate='%{fullData.name}: %{y:.1%}<extra></extra>')
    style(fig,'','Порог оценки модели','Доля')
    fig.add_vline(x=tau, line_color=TEAL, line_dash='dot', name='Выбранный порог', showlegend=True)
    fig.update_xaxes(range=[0,1],tickformat='.0%')
    fig.update_yaxes(tickformat='.0%',range=[0,1.04])
    fig.update_layout(hovermode='x unified',margin_t=28)
    return fig


def miss_chart(groups):
    if groups.empty:
        return empty('')
    counts = groups[['Пропущено', 'Реальные инциденты', 'Инциденты']].astype('Int64')
    counts = counts.to_numpy(dtype=int, na_value=0)
    rates = groups['Доля пропусков'].astype(float)
    bar_lengths = rates.fillna(0)
    labels = [f'{pct(rate)} ({missed}/{total})' if total else '0 из 0 (отсутствуют в указанном периоде)'
              for rate, (missed, total, _) in zip(rates, counts)]
    # Обводка делает нулевой столбец видимым, сохраняя точную длину на оси.
    outline = np.where(bar_lengths.eq(0), 3, 0)
    bar_color = '#C1C1C1'
    fig = go.Figure(go.Bar(
        y=groups['Группа'], x=bar_lengths, orientation='h', marker_color=bar_color,
        marker_line_color=bar_color, marker_line_width=outline,
        text=labels, texttemplate=[label.replace(' в указанном', '<br>в указанном') for label in labels],
        textposition='outside', cliponaxis=False, customdata=counts,
        hovertemplate='%{y}<br>Карточек: %{customdata[2]}'
                      '<br>Пропущено / реальных: %{text}<extra></extra>',
    ))
    style(fig, '', 'Доля пропусков')
    fig.update_xaxes(range=[0, max(.1, min(1, 1.1 * bar_lengths.max()))], tickformat='.0%', nticks=5)
    fig.update_yaxes(autorange='reversed', ticklabelstandoff=8)
    fig.update_layout(margin_t=4, margin_r=107, showlegend=False)
    return fig


def severity_misses(df, tau):
    groups = error_groups(df, tau, 'Критичность').set_index('Группа')
    levels = [*SEVERITY_LEVELS, *groups.index.difference(SEVERITY_LEVELS)]
    return groups.reindex(levels).reset_index()


def enrichment_groups(df, tau):
    labels = {
        (True, True): 'Алерт и вендор',
        (True, False): 'Только алерт',
        (False, True): 'Только вендор',
        (False, False): 'Оба поля пустые',
    }
    filled_fields = zip(df['filled_Алерт'], df['filled_Вендор из сием'])
    grouped = df.assign(enrichment=[labels[state] for state in filled_fields])
    groups = error_groups(grouped, tau, 'enrichment').set_index('Группа')
    return groups.loc[[label for label in labels.values() if label in groups.index]].reset_index()


def daily_misses(df):
    if not df.y_true.eq(1).any():
        return empty('', 'В выбранном срезе нет реальных инцидентов')
    counts = (df.assign(missed=df.result.eq('FN')).groupby('day')
              .agg(missed=('missed', 'sum'), actual=('y_true', 'sum')))
    days = pd.date_range(counts.index.min(), counts.index.max(), freq='D')
    counts = counts.reindex(days, fill_value=0)
    rate = counts.missed / counts.actual.where(counts.actual.gt(0))
    fig = go.Figure(go.Scatter(
        x=days, y=rate, mode='lines+markers', line_color=RED, marker_size=5,
        customdata=counts[['missed', 'actual']].to_numpy(),
        hovertemplate='%{x|%d.%m.%Y}<br>Пропущено %{customdata[0]} из %{customdata[1]}'
                      ' (%{y:.1%})<extra></extra>',
    ))
    style(fig, '', 'Дата создания, UTC', 'Доля пропусков')
    fig.update_xaxes(tickformat='%d.%m', nticks=5)
    fig.update_yaxes(tickformat='.0%', rangemode='tozero')
    fig.update_layout(margin_t=5)
    return fig


def build(dataset="Своя модель · тест", tau=None, start=None, end=None,
          source=ALL, criticality=ALL, incident_type=ALL):
    tau = metadata()["threshold"] if tau is None else float(tau)
    df = model_cards(dataset)
    df = select_model(df,start,end,source,criticality,incident_type)
    m = metrics(df, tau)
    df["result"] = outcomes(df, tau)
    summary = (f"Инцидентов: {m['n']} · Пропущено: {m['fn']} из {m['tp']+m['fn']} реальных "
               f"({pct(m['miss'])}) · Реальных среди тревог: {pct(m['precision'])} · "
               f"Направлено аналитику: {m['tp']+m['fp']} · Порог: {tau:.8f}")
    groups = error_groups(df, tau, "Наименование").sort_values(["Пропущено", "Реальные инциденты"], ascending=False)
    sample_table = df[[c for c in ["EntityId", "team", "y_true", "proba", "result", "Наименование", "Критичность"] if c in df]].copy()
    groups["Доля пропусков"] = groups["Доля пропусков"].map(pct)
    enrich = enrichment_groups(df, tau)
    enrich_fig = miss_chart(enrich)
    enrich["Доля пропусков"] = enrich["Доля пропусков"].map(pct)
    return dict(inference_summary=summary, confusion=agreement_chart(m), tradeoff=workload_chart(df,tau),
                error_rules=miss_chart(severity_misses(df, tau)),
                inference_time=daily_misses(df), enrichment_table=enrich,
                inference_table=sample_table, error_table=groups, enrichment=enrich_fig,
                infer_n=str(m["n"]), infer_miss=pct(m["miss"]),
                infer_missed_count=f"{m['fn']} / {m['tp']+m['fn']}",
                infer_precision=pct(m["precision"]), infer_load=f"{m['tp']+m['fp']} / {m['n']}")


PAGE = "<|part|class_name=dashboard|\n" + intro(
    "Инференс модели", "", """
<|{infer_dataset}|selector|lov={infer_dataset_options}|value_by_id=True|dropdown|label=Модель и выборка|class_name=infer-dataset-select|on_change=on_inference_dataset|>

<|{infer_start}|date|label=Начало|format=dd.MM.yyyy|on_change=on_inference_change|>

<|{infer_end}|date|label=Конец|format=dd.MM.yyyy|on_change=on_inference_change|>

<|{infer_source}|selector|lov={soc_sources}|dropdown|label=Источник|on_change=on_inference_change|>

<|{infer_criticality}|selector|lov={soc_criticalities}|dropdown|label=Критичность|on_change=on_inference_change|>

<|{infer_type}|selector|lov={soc_types}|dropdown|label=Тип инцидента|on_change=on_inference_change|>


<|{infer_tau}|number|min=0|max=1|step=0.01|label=Порог|on_change=on_inference_change|>

<|Сброс|button|on_action=reset_inference_filters|>

""", "infer-toolbar") + "<|part|class_name=kpi-row|\n" + (
    kpi("Доля пропусков (FNR)", "infer_miss", "FN / (TP + FN)", alert=True)
    + kpi("Пропущено/Всего реальных", "infer_missed_count", "Числа в выбранной подборке")
    + kpi("Реальных среди тревог (Precision)", "infer_precision", "TP / (TP + FP)")
    + kpi("Направлено / всех карточек (Workload)", "infer_load", "Нагрузка при выбранном пороге")
) + "|>\n\n<|part|class_name=dashboard-grid inference-evaluation-grid|\n" + (
    panel("Матрица ошибок модели", "confusion")
    + panel("Пропуски и нагрузка (FNR, Workload)", "tradeoff")
    + panel("Пропуски по уровням критичности (FNR)", "error_rules")
    + panel("Доля пропусков по дням (FNR)", "inference_time", classes="span-two")
    + panel("Пропуски по наличию алерта и вендора (FNR)", "enrichment")
) + "|>\n|>\n"
