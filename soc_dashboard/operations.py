"""Рабочая очередь по историческому срезу: SLA, приоритеты и повторные цели."""
import json
import numpy as np
import pandas as pd


def structured_name(value, person=False):
    if value is None or pd.isna(value):
        return 'Не назначен' if person else 'Не указана'
    try:
        item=json.loads(str(value))
        if isinstance(item,dict):
            if person:
                return ' '.join(str(item.get(k,'')).strip() for k in ['lastName','firstName']).strip() or 'Не назначен'
            return str(item.get('name','Не указана'))
    except (ValueError,TypeError):
        pass
    return str(value).strip() or ('Не назначен' if person else 'Не указана')


def prepare_operations(frame):
    """Не используем устаревший статус New из SIEM как статус расследования."""
    df=frame.copy()
    closed=pd.to_datetime(df['Дата закрытия (факт)'],utc=True,errors='coerce')
    taken=pd.to_datetime(df['Дата взятия в работу'],utc=True,errors='coerce')
    snapshot=pd.concat([df.created,closed,taken],ignore_index=True).max()
    active=closed.isna()
    age=(snapshot-df.created).dt.total_seconds()/3600
    df['age_h']=age.where(age.ge(0))
    df['taken_at']=taken
    df['owner']=df['Ответственный'].map(lambda x:structured_name(x,True))
    df['group']=df['Группа устранения'].map(structured_name)
    sla=pd.to_numeric(df['SLA по взятию в работу Атаки'],errors='coerce')/3_600_000
    # В исходном поле миллисекунды. Длительности >30 дней считаем сомнительными.
    valid_sla=sla.gt(0)&sla.le(30*24)
    df['take_sla_h']=sla.where(valid_sla)
    unknown=taken.notna()&taken.lt(df.created)
    df['waiting']=active&taken.isna()
    df['response_overdue']=df.waiting&valid_sla&df.age_h.gt(sla)&~unknown
    df['critical_active']=active&df['Критичность'].eq('Критическая')
    df['high_active']=active&df['Критичность'].isin(['Критическая','Высокая'])
    df['priority']=np.select([df.critical_active,df.response_overdue,
        df.high_active,active],[0,1,2,3],default=4)
    df['action']=np.select([df.critical_active,df.response_overdue,
        df.waiting&df.owner.eq('Не назначен'),df.waiting,active],
        ['Эскалация: критическая','Взять: SLA просрочен','Назначить исполнителя',
         'Взять в работу','Продолжить разбор'],default='Закрыт')
    df['sla_state']=np.select([df.response_overdue,active&unknown,
        df.waiting&~valid_sla,df.waiting,active],
        ['ПРОСРОЧЕНО','Ошибка времени','Нет норматива','Ожидает взятия','В работе'],default='Закрыт')
    df.attrs['snapshot']=snapshot
    return df


def priority_queue(frame,probabilities=None):
    df=frame.copy()
    if probabilities is not None:
        df['risk']=df.EntityId.map(probabilities)
    else:
        df['risk']=np.nan
    # ML не может опустить критическую или просроченную инцидент ниже обычной.
    df=df.sort_values(['priority','response_overdue','risk','age_h','EntityId'],
                     ascending=[True,False,False,False,True],kind='stable',na_position='last')
    columns={'EntityId':'№','Критичность':'Уровень','sla_state':'SLA','action':'Действие','Наименование':'Правило','owner':'Исполнитель'}
    table=df[list(columns)].rename(columns=columns).copy()
    table['Возраст']=df.age_h.map(lambda x:'Нет данных' if pd.isna(x) else f'{x/24:.1f} д' if x>=24 else f'{x:.1f} ч')
    table=table[['№','Уровень','SLA','Возраст','Действие','Правило','Исполнитель']]
    if probabilities is not None:
        table['Риск']=df.risk.map(lambda v:'Нет оценки' if pd.isna(v) else f'{v:.1%}')
    return table.reset_index(drop=True)


def targets(value):
    if value is None or pd.isna(value):
        return set()
    raw=str(value)
    try:
        parsed=json.loads(raw)
        parts=parsed if isinstance(parsed,list) else [raw]
    except (ValueError,TypeError):
        parts=[raw]
    result=set()
    for part in parts:
        for token in str(part).split('|'):
            tokens=[x.strip().lower().rstrip('.') for x in token.split('\\') if x.strip()]
            if tokens and tokens[0] not in ['0.0.0.0','::','-','null','none','[]']:
                result.add(tokens[0])
    return result


def target_counts(frame):
    pairs=[(int(row.EntityId),host) for row in frame[['EntityId','Hostname Цели']].itertuples(index=False)
           for host in targets(row[1])]
    if not pairs:
        return pd.Series(dtype='int64')
    counts=pd.DataFrame(pairs,columns=['id','host']).drop_duplicates().groupby('host').id.nunique()
    return counts.sort_values(ascending=False)


def repeated_targets(frame):
    counts=target_counts(frame)
    return counts[counts.ge(2)]
