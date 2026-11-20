"""Запуск: python -m soc_dashboard.app [--port 5050]."""
import argparse
import math
import json

import pandas as pd
from taipy.gui import Gui, notify
from . import cards, inference
from .data import ALL, date_limits, model_cards, raw_card, soc_cards
from .ml import metadata

navigation = [('/cards', 'Инциденты SOC'), ('/inference', 'Инференс модели')]
chart_config = {'displayModeBar':False,'responsive':True,'displaylogo':False}
soc_sources = [ALL, *sorted(soc_cards()['Источник инцидента'].unique())]
soc_criticalities = [ALL, *sorted(soc_cards()['Критичность'].unique())]
soc_types = [ALL, *sorted(soc_cards()['Тип инцидента'].unique())]
soc_groups = [ALL,*sorted(soc_cards()['group'].unique())]


DEFAULT_INFER_DATASET = 'Своя модель · тест'
infer_dataset_options = [
    (DEFAULT_INFER_DATASET, 'GradientBoosting test (август)'),
    ('Своя модель · сентябрь', 'GradientBoosting inference (сентябрь)'),
]


def soc_defaults():
    start,end = date_limits(soc_cards())
    return dict(soc_start=start,soc_end=end,soc_status='Открыт',soc_source=ALL,
                soc_criticality=ALL,soc_type=ALL,soc_group=ALL)


def infer_defaults(dataset):
    start,end = date_limits(model_cards(dataset))
    return dict(infer_dataset=dataset,infer_start=start,infer_end=end,infer_source=ALL,
                infer_criticality=ALL,infer_type=ALL,infer_tau=metadata()['threshold'])


def soc_result(values):
    return cards.build(source=values['soc_source'],criticality=values['soc_criticality'],
        group=values['soc_group'],start=values['soc_start'],end=values['soc_end'],
        status=values['soc_status'],incident_type=values['soc_type'])


def infer_result(values):
    return inference.build(dataset=values['infer_dataset'],tau=values['infer_tau'],
        start=values['infer_start'],end=values['infer_end'],source=values['infer_source'],
        criticality=values['infer_criticality'],incident_type=values['infer_type'])


SOC_FIELDS = list(soc_defaults())
soc_applied = soc_defaults()
globals().update(soc_applied)
globals().update(soc_result(soc_applied))
infer_applied = infer_defaults(DEFAULT_INFER_DATASET)
INFER_FIELDS = list(infer_applied)
globals().update(infer_applied)
globals().update(infer_result(infer_applied))
soc_detail_open = False
soc_detail_title = ''
soc_detail_action = ''
soc_detail_action_class = 'decision-note'
soc_detail_fields = pd.DataFrame(columns=['Поле', 'Значение'])


def update(state, values):
    with state:
        for name,value in values.items():
            setattr(state,name,value)


def check_dates(start,end):
    if start is None or end is None:
        raise ValueError('Укажите начало и конец периода')
    if pd.Timestamp(start).date() > pd.Timestamp(end).date():
        raise ValueError('Начало периода позже конца')


def on_soc_change(state,var_name,value):
    values = {name:getattr(state,name) for name in SOC_FIELDS}
    values[var_name] = value  # Не зависим от порядка propagate / callback в Taipy.
    try:
        check_dates(values['soc_start'],values['soc_end'])
        result = soc_result(values)
    except ValueError as error:
        notify(state,'warning',str(error))
        update(state,state.soc_applied)
        return
    update(state,{**values,**result,'soc_applied':values.copy()})


def reset_soc_filters(state,id=None,payload=None):
    values = soc_defaults()
    update(state,{**values,**soc_result(values),'soc_applied':values.copy()})


def on_inference_change(state,var_name,value):
    values = {name:getattr(state,name) for name in INFER_FIELDS}
    values[var_name] = value
    try:
        check_dates(values['infer_start'],values['infer_end'])
        tau = float(values['infer_tau'])
        if not math.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError('Порог должен быть числом от 0 до 1')
        values['infer_tau'] = tau
        result = infer_result(values)
    except (TypeError,ValueError) as error:
        notify(state,'warning',str(error))
        update(state,state.infer_applied)
        return
    update(state,{**values,**result,'infer_applied':values.copy()})


def on_inference_dataset(state,var_name,value):
    values = infer_defaults(value)
    update(state,{**values,**infer_result(values),'infer_applied':values.copy()})


def reset_inference_filters(state,id=None,payload=None):
    # Сохраняем выбранную модель/выборку, сбрасываем всё остальное и её порог.
    values = infer_defaults(state.infer_dataset)
    update(state,{**values,**infer_result(values),'infer_applied':values.copy()})


def queue_row_class(state,index,row):
    if row.get('Уровень') == 'Критическая' and row.get('Действие') != 'Закрыт':
        return 'queue-critical'
    return 'queue-overdue' if row.get('SLA') == 'ПРОСРОЧЕНО' else ''


def severity_cell_class(state,value,index,row,column_name):
    return 'critical-text' if 'крит' in str(value).lower() else ''


def readable(value):
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return "Не заполнено"
    try:
        decoded = json.loads(str(value))
        if isinstance(decoded, list):
            return "\n".join(map(str, decoded)) or "Пустой массив"
    except (ValueError, TypeError):
        pass
    return str(value)


def on_queue_row(state,var_name,payload):
    # Taipy передаёт индекс исходной строки, в том числе после сортировки в UI.
    table = state.queue_table
    index = (payload or {}).get('index')
    if index not in table.index:
        return
    selected = table.loc[index]
    entity_id = int(selected['№'])
    raw = raw_card(entity_id)
    op = soc_cards().set_index('EntityId').loc[entity_id]
    values = {
        'Создан (UTC)':op.created.strftime('%d.%m.%Y %H:%M'),
        'Критичность':op['Критичность'],'Возраст на дату среза':selected['Возраст'],
        'SLA взятия':op.sla_state,'Норматив взятия, ч':op.take_sla_h,
        'Дата взятия':op.taken_at,'Исполнитель':op.owner,'Группа':op['group'],
        'Правило':op['Наименование'],'Тип':op['Тип инцидента'],
        'Целевые узлы':raw.get('Hostname Цели'),'Источник':op['Источник инцидента'],
        'Алерт':raw.get('Алерт'),'Команда':raw.get('Команда'),
        'Описание':raw.get('Описание инцидента'),'Рекомендации по плейбуку':raw.get('Рекомендации по плейбуку'),
    }
    fields = pd.DataFrame({'Поле':list(values),'Значение':[readable(v) for v in values.values()]})
    update(state,dict(soc_detail_open=True,
        soc_detail_title=f'Инцидент №{entity_id}',soc_detail_action=selected['Действие'],soc_detail_fields=fields,
        soc_detail_action_class='decision-note critical-text' if 'крит' in selected['Действие'].lower() else 'decision-note'))


def close_soc_detail(state,id=None,payload=None):
    state.soc_detail_open = False


ROOT_PAGE = '''<|part|class_name=header|
<|navbar|lov={navigation}|>

|>
<|content|>

<|{soc_detail_open}|dialog|title={soc_detail_title}|width=90vw|labels=Закрыть|on_action=close_soc_detail|
<|{soc_detail_action}|text|class_name={soc_detail_action_class}|>

<|{soc_detail_fields}|table|page_size=8|editable=False|height=55vh|cell_class_name[Значение]=severity_cell_class|>

|>
'''
# Taipy получает глобальные переменные и обработчики из контекста этого модуля.
gui = Gui(pages={'/': ROOT_PAGE, 'cards': cards.PAGE, 'inference': inference.PAGE}, css_file='style.css')


def create_gui():
    return gui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=5050)
    args = parser.parse_args()
    gui.run(host='127.0.0.1',port=args.port,title='Дашборды',dark_mode=False,
            use_reloader=False,run_browser=False,watermark='',margin='0px',
            stylekit={'color_primary':'#606060','color_secondary':'#FF0000',
                      'color_background_light':'#FFFFFF','color_paper_light':'#FFFFFF'})


if __name__=='__main__':
    main()
