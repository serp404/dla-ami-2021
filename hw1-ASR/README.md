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

## Recommended implementation order

You might be a little intimidated by the number of folders and classes. Try to follow this steps to gradually undestand
the workflow.

1) Test `hw_asr/tests/test_dataset.py`  and `hw_asr/tests/test_config.py` and make sure everythin works for you
2) Implement missing functions to fix tests in  `hw_asr\tests\test_text_encoder.py`
3) Implement missing functions to fix tests in  `hw_asr\tests\test_dataloader.py`
4) Implement functions in `hw_asr\metric\utils.py`
5) Implement missing function to run `train.py` with a baseline model
6) Write your own model and try to overfit it on a single batch
7) ~~Pain and suffering~~ Implement your own models and train them. You've mastered this template when you can tune your
   experimental setup just by tuning `configs.json` file and running `train.py`
8) Don't forget to write a report about your work
9) Get hired by Google the next day

## Before submitting

0) Make sure your projects run on a new machine after complemeting installation guide
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors

   ```shell
   python -m unittest discover hw_asr/tests
   ```

3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.

   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```

4) Use `train.py` for training
