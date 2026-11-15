"""Агрегирование до отрисовки: ограниченное число интервалов, все строки учтены."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .charts import style, empty, bar_marker


def _time_window(timestamps,start,end):
    lo=pd.Timestamp(start).normalize().tz_localize(None) if start is not None else timestamps.min().normalize()
    hi=(pd.Timestamp(end).normalize().tz_localize(None)+pd.Timedelta(days=1)
        if end is not None else timestamps.max().normalize()+pd.Timedelta(days=1))
    if lo>=hi:
        raise ValueError('Начало периода позже конца')
    return lo,hi


def _time_edges(lo,hi,rows):
    # На коротком периоде — часы, на длинном — дни/недели/месяцы.
    # Небольшая учебная выборка получает меньше интервалов, чем полный поток.
    target=min(24,max(6,int(np.sqrt(max(rows,1)))))
    step=(hi-lo).total_seconds()/target
    intervals=[('h',3600,'по часам'),('2h',7200,'каждые 2 часа'),
        ('3h',10800,'каждые 3 часа'),('6h',21600,'каждые 6 часов'),('12h',43200,'каждые 12 часов'),
        ('D',86400,'по дням'),('2D',172800,'каждые 2 дня'),('3D',259200,'каждые 3 дня'),
        ('7D',604800,'по неделям'),('14D',1209600,'по 2 недели'),
        ('MS',2629800,'по месяцам'),('2MS',5259600,'по 2 месяца'),
        ('3MS',7889400,'по кварталам'),('6MS',15778800,'по полугодиям'),('YS',31557600,'по годам')]
    frequency,seconds,label=next((v for v in intervals if v[1]>=step),intervals[-1])
    if frequency=='YS' and step>seconds:
        years=int(np.ceil(step/seconds));frequency=f'{years}YS';label=f'по {years} лет'
    offset=pd.tseries.frequencies.to_offset(frequency)
    anchor=lo.replace(day=1) if frequency.endswith('MS') else lo.replace(month=1,day=1) if frequency.endswith('YS') else lo
    boundaries=pd.date_range(anchor,hi+offset,freq=frequency)
    edges=pd.DatetimeIndex([lo,*boundaries[(boundaries>lo)&(boundaries<hi)],hi])
    return edges,frequency,label


def time_histogram(df,category,colors=None,start=None,end=None,*,show_period_bounds=False,adaptive_scale=False):
    if df.empty and (start is None or end is None):
        return empty('Поток инцидентов')
    timestamps=pd.to_datetime(df.created,utc=True,errors='coerce').dt.tz_localize(None)
    if not timestamps.notna().any() and (start is None or end is None):
        return empty('Поток инцидентов')
    lo,hi=_time_window(timestamps,start,end)
    valid=timestamps.notna()&timestamps.ge(lo)&timestamps.lt(hi)
    edges,frequency,label=_time_edges(lo,hi,int(valid.sum()))
    positions=np.searchsorted(edges.asi8,timestamps.astype('int64').to_numpy(),side='right')-1
    work=df.loc[valid,[category]].copy()
    work['_bin']=positions[valid.to_numpy()]
    counts=work.groupby(['_bin',category],observed=True).size().unstack(fill_value=0)
    counts=counts.reindex(range(len(edges)-1),fill_value=0)
    positive=counts.to_numpy()
    positive=positive[positive>0]
    log_scale=adaptive_scale and positive.size>0 and positive.max()>=1000*positive.min()
    starts,ends=edges[:-1],edges[1:]
    centers=starts+(ends-starts)/2
    widths=(ends-starts).total_seconds()*1000*.9
    custom=[[a.strftime('%d.%m.%Y %H:%M'),(b-pd.Timedelta(nanoseconds=1)).strftime('%d.%m.%Y %H:%M')]
            for a,b in zip(starts,ends)]
    fig=go.Figure()
    for index,column in enumerate(counts):
        marker=bar_marker(column,index)
        if colors and column in colors:
            marker['color']=colors[column]
        values=counts[column].to_numpy()
        fig.add_bar(x=centers,y=np.where(values>0,values-.5,np.nan) if log_scale else values,
                    base=.5 if log_scale else None,width=None if log_scale else widths,
                    name=str(column),marker=marker,
                    customdata=[[*interval,int(value)] for interval,value in zip(custom,values)],
                    hovertemplate='%{customdata[0]} — %{customdata[1]}<br>Инцидентов: %{customdata[2]:,d}<extra>%{fullData.name}</extra>')
    style(fig,'',f'Дата создания, UTC ({label})','Инциденты')
    fig.update_layout(barmode='group' if log_scale else 'stack',bargap=.1)
    if log_scale:
        fig.update_yaxes(type='log',range=[np.log10(.5),np.log10(positive.max())+.12],
                        dtick=1,tickformat=',d',title_text='Инциденты (лог. шкала)')
    else:
        peak=counts.sum(axis=1).max()
        fig.update_yaxes(autorange=True,rangemode='tozero',tickformat=',d',
                         dtick=1 if 0<peak<=10 else None)
    day_ms=86_400_000  # Plotly задаёт шаг временной оси в миллисекундах или календарных месяцах.
    fig.update_xaxes(type='date',range=[lo.isoformat(),hi.isoformat()],autorange=False,
        tickmode='auto',tickformatstops=[
            dict(dtickrange=[None,day_ms],value='%H:%M\n%d.%m'),
            dict(dtickrange=[day_ms,'M1'],value='%d.%m\n%Y'),
            dict(dtickrange=['M1','M12'],value='%m.%Y'),
            dict(dtickrange=['M12',None],value='%Y'),
        ])
    if show_period_bounds:
        last=hi-pd.Timedelta(milliseconds=1)
        middle=lo+(hi-lo)/2
        if hi-lo<=pd.Timedelta(days=2):
            labels=[lo.strftime('%d.%m.%Y<br>%H:%M'),middle.strftime('%H:%M'),last.strftime('%d.%m.%Y<br>%H:%M')]
        else:
            middle=middle.normalize()
            labels=[lo.strftime('%d.%m.%Y'),middle.strftime('%d.%m'),last.strftime('%d.%m.%Y')]
        fig.update_xaxes(range=[lo.isoformat(),last.isoformat()],fixedrange=True,tickmode='array',
                         tickvals=[lo.isoformat(),middle.isoformat(),last.isoformat()],ticktext=labels)
    if work.empty:
        fig.add_annotation(text='За выбранный период инцидентов нет',x=.5,y=.5,
                           xref='paper',yref='paper',showarrow=False)
        if show_period_bounds:
            fig.update_yaxes(visible=False)
    fig.update_layout(meta={'interval':frequency,'rows':len(work),'bins':len(edges)-1,
                            'start':lo.isoformat(),'end_exclusive':hi.isoformat()})
    return fig


def duration_histogram(df):
    valid=df.loc[df.duration_h.notna()]
    if valid.empty:
        return empty('Время решения')
    upper=float(np.log1p(valid.duration_h.max()))
    edges=np.linspace(0,max(upper,1e-6),15)
    centers=(edges[:-1]+edges[1:])/2
    custom=np.column_stack([np.expm1(edges[:-1]),np.expm1(edges[1:])])
    fig=go.Figure()
    for label,group in valid.groupby('verdict_group'):
        counts,_=np.histogram(np.log1p(group.duration_h),bins=edges)
        fig.add_bar(x=centers,y=counts,width=np.diff(edges)*.9,name=label,
                    marker=bar_marker(label),customdata=custom,
                    hovertemplate='%{customdata[0]:.1f}–%{customdata[1]:.1f} ч<br>Инцидентов: %{y}<extra>%{fullData.name}</extra>')
    fig.update_layout(barmode='stack')
    style(fig,'','Время решения, ч · log(1 + ч)','Инциденты')
    ticks=[0,1,24,168,720,8760]
    fig.update_xaxes(tickvals=np.log1p(ticks),ticktext=list(map(str,ticks)))
    fig.update_yaxes(rangemode='tozero',tickformat=',d')
    return fig
