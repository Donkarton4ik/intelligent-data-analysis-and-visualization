import pandas as pd
import pytest

from soc_dashboard import ml
from soc_dashboard.data import between_dates, model_cards


def test_training_split_is_temporal_and_disjoint():
    info=ml.metadata()
    groups=[set(ids) for ids in info['split_ids'].values()]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert info['periods']=={'train':'2025-06','validation':'2025-07','test':'2025-08','fresh':'2025-09'}
    assert 'Вердикт' not in info['input_columns'] and 'Решение' not in info['input_columns']
    assert not any(x in info['input_columns'] for x in ['Ложное срабатывание','False Positive','Статус','Комментарий'])
    test=model_cards('Своя модель · тест')
    assert len(test)==info['test']['n']==9200
    assert test.created.ge('2025-08-01').all() and test.created.lt('2025-09-01').all()


def test_date_range_is_inclusive_of_whole_last_day():
    df=pd.DataFrame({'created':pd.to_datetime(['2025-09-01T00:00:00Z','2025-09-01T23:59:59Z',
                                             '2025-09-02T00:00:00Z'])})
    assert len(between_dates(df,'2025-09-01','2025-09-01'))==2
    with pytest.raises(ValueError):between_dates(df,'2025-09-02','2025-09-01')
