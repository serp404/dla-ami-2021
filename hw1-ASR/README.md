# ASR project barebones

Работа на домашним заданием происходила под ОС Ubuntu 20.01 с использованием Python 3.9, поэтому лучше иметь такую конфигурацию

## Installation guide

Устанавливаем основные библиотеки:

```shell
pip install -r ./requirements.txt
```

Устанавливаем дополнительные библиотеки:

Для `kenlm library`:

```shell
pip install https://github.com/kpu/kenlm/archive/master.zip
```

Для `ctcdecode`:

```shell
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

Если в результате установки последних двух библиотек что-то пошло не так и весь ~~рот~~ экран оказался в C++ ошибках, то возможно поможет эта команда:

```shell
sudo apt-get install python3.9-dev
```

## Testing

Все последующие операции делаем в этой директории:

Веса модели лежат тут: <https://drive.google.com/file/d/1P2N45LCXdv6QajbmaD6ZCvZUm56wvu9O/view?usp=sharing>

Скачиваем веса и перемещаем в тестовую директорию:

```shell
gdown --id 1P2N45LCXdv6QajbmaD6ZCvZUm56wvu9O
mv checkpoint.pth ./default_test_model/
```

Запускаем на тестовых данных:

```shell
python3 test.py -r default_test_model/checkpoint.pth -t test_data/ -o test_output.json
```

В файле `test_output.json` будут лежать предсказания модели

## Brief report

Что сделано:

- Использована модель DeepSpeech
- Добавлен beam-search с lm-fusion
- Достигнут 0.3 CER
- Чтобы воспроизвести обучение модели достаточно запустить `train.py` с конфигом из `default_test_model`
- Изначально пробовал различные рекурентные сети (они показывали многообещающие результаты), но в итоге сел за `DeepSpeech` и ~~обоср~~ засел с ним надолго...

Графики обучения тут:
<https://wandb.ai/serp404/asr_project/reports/DLA-report--VmlldzoxMTY2NjYz?accessToken=0vhiz6hqs3cgyvne3rkuf4zohwmqvolmd383s0mgf25ro22ggrqzsure47ps6rii>
