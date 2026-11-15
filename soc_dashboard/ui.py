"""Крупные показатели и графики без поясняющего текста на обзорном экране."""
import re


def panel(title, figure, caption="", classes="", legend=""):
    # Обоснования выбора графиков находятся в отчёте, не занимают экран.
    return f"""<|part|class_name=panel {classes}|
<|{title}|text|class_name=panel-title|>

{legend}
<|chart|figure={{{figure}}}|height=100%|width=100%|plot_config={{chart_config}}|>
|>

"""


def readonly_columns(columns):
    # Taipy 4.1 в режиме auto_loading учитывает editable отдельно по столбцам.
    flags = '|'.join(f'editable[{name}]=False' for name in columns.split(';'))
    return f'columns={columns}|editable=False|{flags}'


def kpi(label, variable, detail="", alert=False):
    label = re.sub(r'(?i)(крит[а-яё]*)',r'<|\1|text|class_name=critical-text|>',label)
    return f"""<|part|class_name=kpi {'alert' if alert else ''}|
<|{{{variable}}}|text|class_name=kpi-value|>

<|part|class_name=kpi-label|
{label}
|>
|>

"""


def intro(title, subtitle, controls, classes=""):
    return f"""<|part|class_name=toolbar {classes}|
<|part|class_name=page-intro|
<|{title}|text|class_name=page-title|>
|>

<|part|class_name=filters|
{controls}
|>
|>

"""
