# Панель SOC на Taipy

## Запуск

Скачайте `iavd_ml_dataset.parquet` из папки `3.6_Пр_Интеграция_ML` и сохраните его по пути `data/iavd_ml_dataset.parquet` относительно корня репозитория. [Материалы курса на Яндекс Диске](https://disk.yandex.ru/d/waiZmiOzun8EZA)

Понадобятся Python 3.11 и `uv`. Выполните команды из корня проекта.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python -m soc_dashboard.app
```

Если `.venv` уже создана с Python 3.11, начните со второй команды. После запуска откройте [панель SOC](http://127.0.0.1:5050/cards) в браузере.

## Отчёты

В корне проекта лежат отчёты в Markdown.

- [Лабораторная работа 1. Карточки SOC](lab-1.md)
- [Лабораторная работа 2. Инференс модели](lab-2.md)
