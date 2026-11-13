import numpy as np
import pandas as pd
from soc_dashboard import operations
from soc_dashboard.data import soc_cards


def sample():
    df=pd.DataFrame({
        'EntityId':[1,2,3,4,5,6],
        'created':pd.to_datetime(['2025-11-20T10:00:00Z']*6),
        'Дата закрытия (факт)':[None]*5+['2025-11-20T12:00:00Z'],
        'Дата взятия в работу':['2025-11-20T10:01:00Z',None,None,None,'2025-11-20T09:00:00Z',None],
        'Ответственный':[None]*6,'Группа устранения':['{"name":"L1"}']*6,
        'SLA по взятию в работу Атаки':[600000,600000,10800000,0,600000,600000],
        'Критичность':['Критическая','Средняя','Высокая','Средняя','Средняя','Низкая'],
        'Наименование':['rule']*6,'state':['Открыт']*5+['Закрыт'],
    })
    return operations.prepare_operations(df)


def test_snapshot_sla_units_and_unknown_times():
    df=sample().set_index('EntityId')
    assert df.attrs['snapshot']==pd.Timestamp('2025-11-20T12:00:00Z')
    assert df.age_h.eq(2).all()
    assert np.isclose(df.loc[2,'take_sla_h'],1/6)  # 600000 мс = 10 мин.
    assert df.response_overdue.to_dict()=={1:False,2:True,3:False,4:False,5:False,6:False}
    assert df.loc[4,'sla_state']=='Нет норматива'
    assert df.loc[5,'sla_state']=='Ошибка времени'
    assert df.loc[6,'sla_state']=='Закрыт'


def test_low_model_score_never_demotes_critical_or_overdue():
    df=sample()
    risks=pd.Series({1:.001,2:.002,3:.99,4:1.,5:.8,6:1.})
    queue=operations.priority_queue(df,risks)
    assert queue['№'].tolist()[:3]==[1,2,3]
    assert queue['№'].iloc[-1]==6
    assert queue.index.tolist()==list(range(6))


def test_target_normalization_counts_distinct_cards():
    frame=pd.DataFrame({'EntityId':[1,1,2,3],
        'Hostname Цели':[r'\HOST.EXAMPLE\host|\HOST.EXAMPLE\host',r'\HOST.EXAMPLE\host',
                         '["host.example", "0.0.0.0"]',r'\only-once\short']})
    assert operations.repeated_targets(frame).to_dict()=={'host.example':2}
    assert operations.targets('0.0.0.0|::|[]')==set()


def test_operational_default_and_age_chart_reconcile():
    from soc_dashboard import app
    assert app.soc_defaults()['soc_status']=='Открыт'
    result=app.soc_result(app.soc_defaults())
    active=soc_cards()[lambda d:d.state.eq('Открыт')]
    assert len(result['queue_table'])==len(active)==42055
    assert int(result['soc_overdue'])==int(active.response_overdue.sum())
    assert int(result['soc_waiting'])==int(active.waiting.sum())
    assert sum(sum(t.x) for t in result['age_distribution'].data)==len(active)
    targets=result['repeated_targets'].data[0]
    assert list(targets.x)[::-1]==operations.repeated_targets(active).head(5).to_list()
    trace=result['open_types'].data[0]
    assert dict(zip(trace.customdata,trace.x))==active['Тип инцидента'].value_counts().head(5).to_dict()
    assert 'soc_group' in app.SOC_FIELDS and 'soc_verdict' not in app.SOC_FIELDS
