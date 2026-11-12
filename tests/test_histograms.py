import numpy as np
import pandas as pd
import pytest
from soc_dashboard.histograms import time_histogram, duration_histogram
from soc_dashboard.charts import RED, TEAL
from soc_dashboard.data import soc_cards


@pytest.mark.parametrize('days,frequency',[(10,'2D'),(100,'14D'),(600,'MS')])
def test_time_histogram_aggregates_without_losing_rows(days,frequency):
    frame=pd.DataFrame({'created':pd.date_range('2024-01-01',periods=days,freq='D',tz='UTC'),
                        'source':['SIEM']*days})
    fig=time_histogram(frame,'source',{})
    assert all(trace.type=='bar' for trace in fig.data)
    assert sum(np.sum(trace.y) for trace in fig.data)==days
    assert fig.layout.meta['interval']==frequency
    assert max(len(trace.x) for trace in fig.data)<=35


def test_duration_histogram_includes_zero_and_last_bin():
    frame=pd.DataFrame({'duration_h':[0.,1.,24.,720.,None],
                        'verdict_group':['Инцидент','Инцидент','Ложное срабатывание','Инцидент','Нет вердикта']})
    fig=duration_histogram(frame)
    assert sum(np.sum(trace.y) for trace in fig.data)==4
    assert all(trace.type=='bar' and len(trace.x)==14 for trace in fig.data)


def test_empty_histograms():
    assert len(time_histogram(pd.DataFrame(), 'unused', {}).data)==0
    frame=pd.DataFrame({'duration_h':pd.Series(dtype=float),'verdict_group':pd.Series(dtype=str)})
    assert len(duration_histogram(frame).data)==0


def test_targets_chart_shows_five_most_frequent_named_targets():
    from soc_dashboard.cards import targets_chart
    hosts=[[f'host{i}']*i for i in range(1,8)]
    frame=pd.DataFrame({'Hostname Цели':sum(hosts,[])})
    frame['EntityId']=frame.index
    trace=targets_chart(frame).data[0]
    assert dict(zip(trace.y,trace.x))=={f'host{i}':i for i in range(3,8)}
    assert trace.y[-1]=='host7'
    assert not targets_chart(frame.iloc[:1]).data


def test_period_histogram_covers_whole_selection_not_last_30_days():
    from soc_dashboard import cards
    from soc_dashboard.data import between_dates
    selected=between_dates(soc_cards(),'2025-06-01','2025-09-30')
    selected=selected[selected.state.eq('Открыт')]
    result=cards.build(start='2025-06-01',end='2025-09-30',status='Открыт')
    fig=result['timeline']
    assert sum(sum(trace.y) for trace in fig.data)==len(selected)
    assert fig.layout.meta['rows']==len(selected)
    assert fig.layout.xaxis.range==('2025-06-01T00:00:00','2025-09-30T23:59:59.999000')
    assert (fig.layout.xaxis.ticktext[0],fig.layout.xaxis.ticktext[-1])==('01.06.2025','30.09.2025')


def test_period_histogram_shows_small_counts_on_a_wide_range():
    dates=pd.to_datetime(['2024-04-12']*2000+['2025-11-20'],utc=True)
    frame=pd.DataFrame({'created':dates,'kind':'signal'})
    fig=time_histogram(frame,'kind',start='2024-04-12',end='2025-11-20',
                       show_period_bounds=True,adaptive_scale=True)
    trace=fig.data[0]
    assert fig.layout.yaxis.type=='log'
    assert trace.y[-1]+trace.base==1
    assert sum(int(row[2]) for row in trace.customdata)==len(frame)
    assert fig.layout.xaxis.ticktext[-1]=='20.11.2025'


@pytest.mark.parametrize('days,frequency',[(1,'h'),(2,'2h')])
def test_short_period_uses_intraday_bins_and_exact_window(days,frequency):
    start=pd.Timestamp('2025-11-01',tz='UTC')
    frame=pd.DataFrame({'created':pd.date_range(start,periods=days*1440,freq='min'),'kind':'signal'})
    last=start+pd.Timedelta(days=days-1)
    fig=time_histogram(frame,'kind',start=start,end=last)
    assert fig.layout.meta['interval']==frequency
    assert len(fig.data[0].y)==24
    assert sum(fig.data[0].y)==len(frame)
    assert pd.Timestamp(fig.layout.xaxis.range[0])==start.tz_localize(None)
    assert pd.Timestamp(fig.layout.xaxis.range[1])==(start+pd.Timedelta(days=days)).tz_localize(None)
    assert fig.layout.yaxis.autorange


def test_sparse_sample_is_grouped_without_losing_zero_intervals():
    dates=['2025-09-02','2025-09-03','2025-09-04','2025-09-05',
           '2025-09-25','2025-09-26','2025-09-28','2025-09-30']
    frame=pd.DataFrame({'created':pd.to_datetime(dates,utc=True),'kind':'signal'})
    fig=time_histogram(frame,'kind',start='2025-09-01',end='2025-09-30')
    assert 4<=len(fig.data[0].y)<=8
    assert sum(fig.data[0].y)==len(frame)
    assert 0 in fig.data[0].y  # Реальный перерыв сохраняется нулевым.
    assert fig.layout.meta['end_exclusive']=='2025-10-01T00:00:00'


def test_time_window_excludes_next_midnight_and_handles_empty_result():
    frame=pd.DataFrame({'created':pd.to_datetime(['2025-09-01T00:00:00Z','2025-09-01T23:59:59Z',
                                                '2025-09-02T00:00:00Z']),'kind':'signal'})
    fig=time_histogram(frame,'kind',start='2025-09-01',end='2025-09-01')
    assert sum(fig.data[0].y)==2
    empty=time_histogram(frame.iloc[:0],'kind',start='2025-09-01',end='2025-09-01')
    assert not empty.data and empty.layout.meta['rows']==0
    assert pd.Timestamp(empty.layout.xaxis.range[0])==pd.Timestamp('2025-09-01')


def test_severity_graph_counts_unknown_levels_and_places_values_outside():
    from soc_dashboard.cards import severity_chart
    frame=pd.DataFrame({'Критичность':['Критическая','Высокая','Высокая','Не указано']})
    fig=severity_chart(frame)
    trace=fig.data[0]
    counts=dict(zip(trace.customdata,trace.x))
    assert counts=={'Не указано':1,'Низкая':0,'Средняя':0,'Высокая':2,'Критическая':1}
    assert sum(trace.x)==len(frame)
    assert trace.textposition=='outside' and not trace.cliponaxis
    assert dict(zip(trace.customdata,trace.marker.color))=={
        'Не указано':TEAL,'Низкая':TEAL,'Средняя':TEAL,'Высокая':TEAL,'Критическая':RED}
