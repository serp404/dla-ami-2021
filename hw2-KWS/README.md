# KWS homework

Работа на домашним заданием происходила под ОС Ubuntu 20.01 с использованием Python 3.9, поэтому лучше иметь такую конфигурацию

## Weights

Веса полученных моделей лежат тут: <https://drive.google.com/file/d/12yjwVFVodefysSVYnmPKaa6aPlg30Hmv/view?usp=sharing>

Скачивать и распаковать веса можно так:

```shell
gdown --id 12yjwVFVodefysSVYnmPKaa6aPlg30Hmv
tar -xvf weights.tar
```

Всего в архиве лежат веса 7 моделей:

1. `base_model.pth` - веса обученной базовой модели

2. `simple_dist_model.pth` - веса самой первой и самой просто дистиллированной модели

3. `dist_model.pth` - веса второй версии дистилляции

4. `dist_hard_model.pth` - веса самой сильно дистиллированной модели

5. `quant_base_model.pth` - квантизованная базовая модель

6. `quant_dist_model.pth` - квантизованная модель `dist_model.pth`

7. `quant_dist_hard_model.pth` - квантизованная `dist_hard_model.pth` (самая лучшая из всех)

## Results

1. Compression rate: 17.90901207804274
2. Speedup rate: 12.673832790445168
