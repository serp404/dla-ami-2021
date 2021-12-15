# NV homework

## Installation guide

Запустить `init.sh`. Он должен установить требуемую версию `pytorch`, скачать и распаковать `LJSpeech` в нужную директорию, запустить установку всех пакетов из `requirements.txt`:

```shell
sh init.sh
```

## Checkpoint

Чекпоинт к домашнему заданию (переобучить генератор на одном батче) реализован в файле `dev.ipynb`. В нем находится кодЮ реализующий обучение, а также график падающей ошибки.

Предсказанное переобученным генератором аудио можно найти в на гугл-диске:

<https://drive.google.com/drive/folders/1mrN9o8M3AauKIYM5gvyMKWpK35LvSBPn?usp=sharing>
