from types import SimpleNamespace
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


class State(SimpleNamespace):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_change_does_not_depend_on_taipy_propagation_order():
    from soc_dashboard import app
    state=State(**app.soc_defaults())
    app.on_soc_change(state,'soc_source','definitely-nonexistent')
    assert state.soc_n=='0'
    app.reset_soc_filters(state)
    state=State(**app.infer_defaults('Своя модель · тест'))
    app.on_inference_change(state,'infer_tau',1.)
    assert state.infer_tau==1. and state.infer_load=='0 / 9200'
    assert state.confusion.data[0].customdata[0,0]==1528
    assert state.tradeoff.data[0].y[-1]==1.
    marker, = state.tradeoff.layout.shapes
    assert marker.x0 == marker.x1 == state.infer_tau
    assert marker.showlegend and marker.name == 'Выбранный порог' and marker.line.dash == 'dot'
    app.on_inference_change(state, 'infer_tau', 0.)
    severity = state.error_rules.data[0]
    assert list(severity.x) == [0, 0, 0, 0]
    assert severity.text[1:3] == ('0.0% (0/556)', '0.0% (0/972)')


def test_inference_opens_on_graphs_with_the_independent_test():
    from soc_dashboard import app
    assert app.infer_dataset=='Своя модель · тест'
    matrix = app.confusion.data[0]
    assert matrix.customdata[0, 0] == 101
    assert [[cell.split('<br>')[2] for cell in row] for row in matrix.text] == [['FN', 'TP'], ['TN', 'FP']]
    miss_curve=app.tradeoff.data[0]
    current=next(i for i,x in enumerate(miss_curve.x) if x==app.infer_tau)
    assert miss_curve.y[current]==pytest.approx(101/1528)
    assert all(figure.data for figure in (app.error_rules,app.inference_time,app.enrichment))
    daily = app.inference_time.data[0]
    assert pd.DatetimeIndex(daily.x).equals(pd.date_range('2025-08-01', '2025-08-31'))
    assert daily.customdata.sum(axis=0).tolist() == [101, 1528]
    severity = app.error_rules.data[0]
    assert tuple(severity.y) == ('Критическая', 'Высокая', 'Средняя', 'Низкая')
    assert severity.x[0] == severity.x[-1] == 0
    assert severity.text[0] == severity.text[-1] == '0 из 0 (отсутствуют в указанном периоде)'
    assert list(severity.marker.line.width) == [3, 0, 0, 3]
    enrichment = app.enrichment.data[0]
    assert tuple(enrichment.y) == ('Алерт и вендор', 'Только алерт', 'Оба поля пустые')
    assert enrichment.customdata.tolist() == [[73, 1033, 6121], [0, 0, 98], [28, 495, 2981]]


def test_invalid_date_edit_restores_last_applied_filters(monkeypatch):
    from soc_dashboard import app
    notices=[]
    monkeypatch.setattr(app,'notify',lambda *args:notices.append(args[2]))
    state=State(**app.soc_defaults())
    app.reset_soc_filters(state)
    previous=state.soc_start
    state.soc_start=pd.Timestamp('2030-01-01')
    app.on_soc_change(state,'soc_start',state.soc_start)
    assert state.soc_start==previous and state.soc_n=='42 055'
    state=State(**app.infer_defaults('Своя модель · тест'))
    app.reset_inference_filters(state)
    state.infer_tau=float('nan')
    app.on_inference_change(state,'infer_tau',state.infer_tau)
    assert state.infer_tau==ml.metadata()['threshold']
    assert len(notices)==2


def test_taipy_http_and_socket_connection(caplog):
    from soc_dashboard.app import create_gui
    gui = create_gui()
    gui.run(run_server=False, single_client=True)
    app = gui.get_flask_app()
    with app.test_client() as client:
        for path in ["/", "/cards", "/inference"]:
            assert client.get(path).status_code == 200
        css = client.get("/style.css")
        assert css.status_code == 200 and b".dashboard-grid" in css.data
        # Rendering endpoints exercise variable and callback binding, unlike
        # the HTTP shell, which returns 200 even if every graph is unbound.
        for page in ["cards", "inference", "TaiPy_root_page"]:
            response = client.get(f"/taipy-jsx/{page}")
            assert response.status_code == 200
            jsx=response.get_json()["jsx"]
            assert ("Dialog" if page=="TaiPy_root_page" else "Chart") in jsx
        assert "is not defined" not in caplog.text
        assert "is not a function" not in caplog.text
        assert "unmatched closing tag" not in caplog.text
    # Regression: Flask 3.1.3 breaks the Socket.IO session setter used by Taipy 4.1.1.
    socket = app.extensions["socketio"].test_client(app)
    assert socket.is_connected()
    socket.disconnect()


def test_switch_and_reset_inference_dataset():
    from soc_dashboard import app

    state = State(**app.infer_defaults(app.DEFAULT_INFER_DATASET))
    app.on_inference_dataset(state, 'infer_dataset', 'Своя модель · сентябрь')
    assert state.infer_n == '7885'
    assert state.infer_miss == '13.7%'
    state.infer_tau = 1
    state.infer_source = 'not-a-source'
    app.reset_inference_filters(state)
    assert state.infer_dataset == 'Своя модель · сентябрь'
    assert state.infer_n == '7885'
    assert state.infer_tau == ml.metadata()['threshold']
