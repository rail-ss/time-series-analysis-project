
"""DL-модели временных рядов через NeuralForecast.

Этот файл пока содержит безопасный шаблон. Конкретные параметры мы будем подбирать
после анализа выбранного ряда: длины истории, сезонности, горизонта прогноза и объема данных.
"""

from __future__ import annotations


def get_default_neuralforecast_models(horizon: int, input_size: int, max_steps: int = 200):
    """Возвращает базовый набор DL-моделей.

    Модели:
    - MLP: простая нейросетевая база;
    - LSTM: рекуррентная модель для последовательностей;
    - NHITS: сильная современная модель для прогнозирования временных рядов.
    """
    try:
        from neuralforecast.models import MLP, LSTM, NHITS
    except ImportError as exc:
        raise ImportError("Установите neuralforecast: pip install neuralforecast") from exc

    models = [
        MLP(h=horizon, input_size=input_size, max_steps=max_steps),
        LSTM(h=horizon, input_size=input_size, max_steps=max_steps),
        NHITS(h=horizon, input_size=input_size, max_steps=max_steps),
    ]
    return models


def run_neuralforecast_models(df_train, horizon: int, freq: str, input_size: int, max_steps: int = 200):
    """Обучает DL-модели и возвращает прогноз.

    Ожидаемый формат входа: unique_id, ds, y.
    """
    try:
        from neuralforecast import NeuralForecast
    except ImportError as exc:
        raise ImportError("Установите neuralforecast: pip install neuralforecast") from exc

    models = get_default_neuralforecast_models(horizon, input_size, max_steps=max_steps)
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=df_train)
    return nf.predict()
