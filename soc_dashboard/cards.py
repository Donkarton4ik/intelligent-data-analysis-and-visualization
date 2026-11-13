"""Инциденты SOC: приоритетная очередь дежурного по историческому срезу."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from textwrap import wrap
from html import escape
from .data import soc_cards,select_soc,ALL,SEVERITY_LEVELS
from .charts import style,empty,horizontal,bar_marker,critical_label,RED,TEAL
from .analysis import pct,divide
from .operations import priority_queue,repeated_targets
from .ui import intro,kpi,panel,readonly_columns
from .histograms import time_histogram


def age_chart(active):
    labels=['До 1 ч','1–4 ч','4–24 ч','1–3 суток','Более 3 суток']
    bins=pd.cut(active.age_h,bins=[-np.inf,1,4,24,72,np.inf],labels=labels,right=False)
    fig=go.Figure()
    short={'Критическая':'Критич.','Высокая':'Высок.','Средняя':'Средн.','Низкая':'Низк.'}
    for level in short:
        counts=bins[active['Критичность'].eq(level)].value_counts().reindex(labels,fill_value=0)
        fig.add_bar(x=counts.values[::-1],y=labels[::-1],orientation='h',
                    name=critical_label(short[level]) if level=='Критическая' else short[level],
                    marker=bar_marker(level),hovertemplate='%{y}: %{x}<extra>'+level+'</extra>')
    fig.update_layout(barmode='stack')
    style(fig,'','Открытые инциденты')
    fig.update_layout(margin_t=34,legend=dict(font_size=14))
    return fig


def severity_chart(active):
    levels=list(SEVERITY_LEVELS)
    levels+=sorted(set(active['Критичность'].unique())-set(levels))
    counts=active['Критичность'].value_counts().reindex(levels,fill_value=0)
    if active.empty:
        return empty('Критичность')
    labels=[critical_label(level) if level=='Критическая' else level for level in levels]
    colors=[RED if level=='Критическая' else TEAL for level in levels]
    fig=horizontal(labels,counts.to_numpy(),'Критичность',color=colors)
    fig.data[0].customdata=levels[::-1]
    fig.data[0].hovertemplate='%{customdata}<br>Инцидентов: %{x:,d}<extra></extra>'
    fig.update_layout(margin_t=6)
    return fig


def targets_chart(active):
    counts=repeated_targets(active).head(5)
    if counts.empty:
        return empty('Повторные инциденты по узлам', 'В выбранном срезе нет узлов с повторными инцидентами')
    labels=[escape(host) for host in counts.index]
    ticktext=[escape(host if len(host)<=20 else host[:19]+'…') for host in counts.index]
    fig=horizontal(labels,counts.to_numpy(),'Повторные инциденты по узлам',unit='Открытые инциденты')
    fig.update_yaxes(tickvals=labels,ticktext=ticktext)
    fig.data[0].hovertemplate='%{y}<br>Открытых инцидентов: %{x:,d}<extra></extra>'
    fig.update_layout(margin_t=6)
    return fig


def build(month=ALL,source=ALL,criticality=ALL,verdict=ALL,start=None,end=None,status=ALL,incident_type=ALL,group=ALL):
    full=soc_cards()
    df=select_soc(full,month,source,criticality,verdict,start,end,status,incident_type,group)
    active=df[df.state.eq('Открыт')]
    queue=priority_queue(df)
    queue['Тип']=df.set_index('EntityId').loc[queue['№'],'Тип инцидента'].to_numpy()
    # Дата среза фиксирована в выгрузке, возраст не растёт вместе с текущими часами.
    snapshot=full.attrs['snapshot']
    flow=df.copy()
    flow['flow_kind']=np.where(flow['Критичность'].eq('Критическая'),'Критические','Остальные')
    timeline=time_histogram(flow,'flow_kind',start=start,end=end,show_period_bounds=True,adaptive_scale=True)
    for trace in timeline.data:
        trace.marker=bar_marker('Критическая' if trace.name=='Критические' else 'Низкая')
        if trace.name=='Критические':
            trace.name=critical_label(trace.name)
    targets_plot=targets_chart(active)
    types=active['Тип инцидента'].value_counts().head(5)
    labels=['<br>'.join(escape(line) for line in wrap(name,width=38)) for name in types.index]
    types_plot=horizontal(labels,types.values,'Типы открытых инцидентов')
    if len(types):
        types_plot.data[0].customdata=list(types.index[::-1])
        types_plot.data[0].hovertemplate='%{customdata}<br>Инцидентов: %{x:,d}<extra></extra>'
        types_plot.update_layout(showlegend=False,margin_t=6)
    known=df.label.notna().sum()
    duration=df.duration_h.dropna()
    rules=df.groupby('Наименование').agg(Инциденты=('EntityId','size'),Размечено=('label','count')).reset_index()
    return dict(queue_table=queue,queue_selected=[],age_distribution=age_chart(active) if len(active) else empty('Возраст'),
        repeated_targets=targets_plot,open_types=types_plot,severity_distribution=severity_chart(active),timeline=timeline,
        soc_n=f'{len(df):,}'.replace(',',' '),soc_open=f'{len(active):,}'.replace(',',' '),
        soc_high=str(int(active.high_active.sum())),soc_waiting=str(int(active.waiting.sum())),
        soc_overdue=str(int(active.response_overdue.sum())),
        soc_critical_open=str(int(active.critical_active.sum())),
        soc_summary=f'Инцидентов: {len(df)} · открыто: {len(active)}',
        soc_noise=pct(divide(df.label.eq(0).sum(),known)),soc_known=pct(divide(known,len(df))),
        soc_duration=f'{duration.median():.1f} ч' if len(duration) else '—',rules_table=rules,
        soc_period=f'Срез: {snapshot:%d.%m.%Y %H:%M} UTC')


PAGE='<|part|class_name=dashboard|\n'+intro('Инциденты SOC','',"""
<|{soc_start}|date|label=Начало|format=dd.MM.yyyy|on_change=on_soc_change|>

<|{soc_end}|date|label=Конец|format=dd.MM.yyyy|on_change=on_soc_change|>

<|{soc_status}|selector|lov=Все;Открыт;Закрыт|dropdown|label=Статус|on_change=on_soc_change|>

<|{soc_source}|selector|lov={soc_sources}|dropdown|label=Источник|on_change=on_soc_change|>

<|{soc_criticality}|selector|lov={soc_criticalities}|dropdown|label=Критичность|on_change=on_soc_change|>

<|{soc_type}|selector|lov={soc_types}|dropdown|label=Тип инцидента|on_change=on_soc_change|>

<|{soc_group}|selector|lov={soc_groups}|dropdown|label=Группа|on_change=on_soc_change|>

<|Сброс|button|on_action=reset_soc_filters|>
""",'soc-toolbar')+'<|part|class_name=kpi-row|\n'+(
    kpi('Открытые инциденты','soc_open')
    +kpi('Высокая и критическая','soc_high')
    +kpi('Не взяты в работу','soc_waiting')
    +kpi('Просрочено взятие в работу','soc_overdue',alert=True)
)+'|>\n\n<|part|class_name=dashboard-grid operations-grid soc-operations-grid|\n'+f"""
<|part|class_name=panel queue-panel primary-queue|
<|Очередь разбора: приоритет и срок|text|class_name=panel-title|>

<|{{queue_table}}|table|{readonly_columns('№;Тип;Уровень;Возраст;SLA;Действие')}|auto_loading=True|page_size=50|size=small|height=100%|row_class_name=queue_row_class|cell_class_name[Уровень]=severity_cell_class|cell_class_name[Действие]=severity_cell_class|number_format=%d|>
|>

"""+(
    panel('Критичность открытых инцидентов','severity_distribution','','severity-panel')
    +panel('Типы открытых инцидентов: топ-5','open_types','','types-panel')
    +panel('Повторные инциденты по узлам: топ-5','repeated_targets','','targets-panel')
    +panel('Число инцидентов за рассматриваемый период','timeline','','period-panel')
)+"""|>
|>

"""
