"""Запуск: python -m soc_dashboard.app [--port 5050]."""
import argparse

import pandas as pd
from taipy.gui import Gui, notify
from . import cards
from .data import ALL, date_limits, soc_cards

navigation = [('/cards', 'Инциденты SOC')]
chart_config = {'displayModeBar':False,'responsive':True,'displaylogo':False}
soc_sources = [ALL, *sorted(soc_cards()['Источник инцидента'].unique())]
soc_criticalities = [ALL, *sorted(soc_cards()['Критичность'].unique())]
soc_types = [ALL, *sorted(soc_cards()['Тип инцидента'].unique())]
soc_groups = [ALL,*sorted(soc_cards()['group'].unique())]


def soc_defaults():
    start,end = date_limits(soc_cards())
    return dict(soc_start=start,soc_end=end,soc_status='Открыт',soc_source=ALL,
                soc_criticality=ALL,soc_type=ALL,soc_group=ALL)


def soc_result(values):
    return cards.build(source=values['soc_source'],criticality=values['soc_criticality'],
        group=values['soc_group'],start=values['soc_start'],end=values['soc_end'],
        status=values['soc_status'],incident_type=values['soc_type'])


SOC_FIELDS = list(soc_defaults())
soc_applied = soc_defaults()
globals().update(soc_applied)
globals().update(soc_result(soc_applied))


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


def queue_row_class(state,index,row):
    if row.get('Уровень') == 'Критическая' and row.get('Действие') != 'Закрыт':
        return 'queue-critical'
    return 'queue-overdue' if row.get('SLA') == 'ПРОСРОЧЕНО' else ''


def severity_cell_class(state,value,index,row,column_name):
    return 'critical-text' if 'крит' in str(value).lower() else ''


ROOT_PAGE = '''<|part|class_name=header|
<|navbar|lov={navigation}|>

|>
<|content|>
'''
# Taipy получает глобальные переменные и обработчики из контекста этого модуля.
gui = Gui(pages={'/': ROOT_PAGE, 'cards': cards.PAGE}, css_file='style.css')


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
