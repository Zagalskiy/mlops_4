# mlops_4
MLOps III семестр. Практическое задание №4

Для автоматизации процесса машинного обучения я выбрал Airflow и MLFlow.

Для анализа я выбрал временной ряд (далее - ВР) с количеством упоминаний компании Google (интервал 5 минут) в социальной сети Х*.

Эксперимент заключается в сравнении метрики sMAPE для следующих методов предсказания временных рядов:
- наивное сезонное предсказание с недельной сезонностью;
- предсказание двойным экспоненциальным сглаживанием с учётом тренда;
- предсказание тройным экспоненциальным сглаживанием с учётом тренда и сезонности.
Скриншоты результатов представлены в папке /screenshots.

*Запрещена в РФ; принадлежит корпорации Meta, которая признана в РФ экстремистской.
