# Технологии машинного обучения

**Студент:** Чернев Николай Андреевич  
**Группа:** ИУ5-64Б  
**Курс:** Технологии машинного обучения  
**Учебный год:** 2025/2026

Репозиторий содержит выполненные лабораторные работы, рубежные контроли и НИРС по курсу «Технологии машинного обучения». В работах рассматриваются полный цикл подготовки данных, обучение и сравнение моделей машинного обучения, методы оптимизации, прогнозирование временных рядов и обучение без учителя.

## Содержание

- [Лабораторные работы](#лабораторные-работы)
- [Рубежные контроли](#рубежные-контроли)
- [НИРС](#нирс)
- [Использованный стек](#использованный-стек)
- [Установка и запуск](#установка-и-запуск)
- [Структура проекта](#структура-проекта)
- [Статус проекта](#статус-проекта)

## Лабораторные работы

| № | Тема | Код | Отчёт | Методические материалы |
|---:|---|---|---|---|
| 1 | Разведочный анализ данных. Исследование и визуализация данных | [lab1.ipynb](./lab1/lab1.ipynb) | [PDF](./lab1/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%961%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO__EDA_VISUALIZATION) |
| 2 | Обработка пропусков, кодирование категориальных признаков и масштабирование | [lab2.ipynb](./lab2/lab2.ipynb) | [PDF](./lab2/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%962%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO__MISSING) |
| 3 | Подготовка выборки, кросс-валидация и подбор гиперпараметров: KNN | [lab3.ipynb](./lab3/lab3.ipynb) | [PDF](./lab3/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%963%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO__KNN) |
| 4 | Линейные модели, SVM и деревья решений | [lab4.ipynb](./lab4/lab4.ipynb) | [PDF](./lab4/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%964%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_TREES) |
| 5 | Ансамбли моделей машинного обучения. Часть 1 | [lab5.ipynb](./lab5/lab5.ipynb) | [PDF](./lab5/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%965%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_ENSEMBLES_1) |
| 6 | Введение в методы оптимизации | [lab6.py](./lab6/lab6.py) | [PDF](./lab6/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%966%20%D0%A2%D0%9C%D0%9E.pdf) | [Материалы](https://github.com/ugapanyuk/courses_current/blob/main/pres/tmo/optim/lab_opt.pdf) |
| 7 | Ансамбли моделей машинного обучения. Часть 2 | [lab7.ipynb](./lab7/lab7.ipynb) | [PDF](./lab7/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%967%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_ENSEMBLES_2) |
| 8 | Анализ и прогнозирование временного ряда | [lab8.ipynb](./lab8/lab8.ipynb) | [PDF](./lab8/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%968%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_TIMESERIES) |
| 9 | Методы обучения без учителя | [lab9.ipynb](./lab9/lab9.ipynb) | [PDF](./lab9/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%9B%D0%A0%20%E2%84%969%20%D0%A2%D0%9C%D0%9E.pdf) | [Задание](https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_CLUSTER) |

## Рубежные контроли

- **РК №1** — [ноутбук с решением](./rk1/rk1.ipynb); [задания](https://github.com/ugapanyuk/courses_current/wiki/TMO_RK_1).
- **РК №2** — [результаты и материалы](./rk2/); [архив задания](https://github.com/ugapanyuk/courses_current/blob/main/pres/concept_map_task.zip).

## НИРС

Исследование по прогнозированию годовой стоимости медицинского страхования на основе данных пациента.

- [Основной ноутбук](./nirs/nirs.ipynb)
- [Отчёт PDF](./nirs/%D0%A7%D0%B5%D1%80%D0%BD%D0%B5%D0%B2%20%D0%98%D0%A35-64%D0%91%20%D0%A2%D0%9C%D0%9E%20%D0%9D%D0%98%D0%A0%D0%A1.pdf)
- [Документация исследования](./nirs/documentation.md)
- [Streamlit-приложение](./nirs/app.py)
- [Датасет](./nirs/medical_insurance_cost_dataset.csv)
- [Типовое задание](https://github.com/ugapanyuk/courses_current/wiki/TMO_NIRS), [опорный пример](https://nbviewer.jupyter.org/github/ugapanyuk/courses_current/blob/main/notebooks/ml_project_example/project_classification_regression.ipynb)

В НИРС сравниваются линейные и регуляризованные модели (`LinearRegression`, `Ridge`, `Lasso`), дерево решений, KNN, `SVR`, случайный лес, градиентный и адаптивный бустинг. Для `RandomForest`, `GradientBoosting` и `Ridge` выполнен подбор гиперпараметров через `GridSearchCV` с 5-блочной кросс-валидацией.

## Использованный стек

- **Python и Jupyter Notebook** — выполнение и документирование экспериментов.
- **NumPy, Pandas** — вычисления, загрузка, очистка и анализ табличных данных.
- **Matplotlib, Seaborn** — графики распределений, корреляций, результатов моделей и ошибок.
- **scikit-learn:** `train_test_split`, `KFold`, `GridSearchCV`, `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `OneHotEncoder`, `LabelEncoder`; линейные модели, `KNeighbors`, `SVM`, деревья решений, `Bagging`, `Random Forest`, `Extra Trees`, `AdaBoost`, `Gradient Boosting`, `Stacking`, `MLPRegressor`; `K-Means`, `DBSCAN`, спектральная и агломеративная кластеризация; `PCA`, `t-SNE`; метрики `MAE`, `MSE`, `RMSE`, `R²`, `ARI`, Silhouette и Davies–Bouldin.
- **SciPy** — численная условная оптимизация методом SLSQP.
- **statsmodels** — модель `ARIMA(5, 1, 2)` и анализ ACF/PACF.
- **gplearn** — символьная регрессия на основе genetic programming.
- **Keras / TensorFlow** — LSTM с оптимизатором Adam, Dropout и EarlyStopping.
- **Streamlit, Pickle** — интерактивное приложение НИРС и загрузка сохранённых модели/масштабировщика.
- **LaTeX** — подготовка отчётов в PDF.

## Установка и запуск

В репозитории нет файла `requirements.txt` или другого манифеста зависимостей. Для запуска ноутбуков потребуется окружение с библиотеками, перечисленными в разделе [«Использованный стек»](#использованный-стек).

### Ноутбуки и лабораторная работа №6

Откройте нужный `.ipynb` в Jupyter или JupyterLab и выполните ячейки последовательно. Скрипт лабораторной работы №6 запускается из её каталога:

```bash
cd lab6
python lab6.py
```

### Streamlit-приложение НИРС

```bash
cd nirs
streamlit run app.py
```

Приложение использует датасет `medical_insurance_cost_dataset.csv`, а при наличии файлов `rf_model.pkl` и `scaler.pkl` может загрузить сохранённые объекты.

## Примеры использования

- Провести EDA и визуализировать зависимости признаков в [лабораторной работе №1](./lab1/lab1.ipynb).
- Сравнить способы кодирования и масштабирования в [лабораторной работе №2](./lab2/lab2.ipynb).
- Подобрать `K` для KNN и выполнить кросс-валидацию в [лабораторной работе №3](./lab3/lab3.ipynb).
- Сопоставить ARIMA, символьную регрессию и LSTM в [лабораторной работе №8](./lab8/lab8.ipynb).
- Запустить интерактивный прогноз стоимости страхования через [приложение НИРС](./nirs/app.py).

## Структура проекта

```text
ML_sem6/
├── lab1/ … lab9/     # ноутбуки, исходные материалы, изображения и отчёты
├── rk1/              # Рубежный контроль №1
├── rk2/              # Материалы и результаты Рубежного контроля №2
├── nirs/             # НИРС, датасет, модель и Streamlit-приложение
├── bmstu_logo.jpg    # Изображение, используемое в материалах отчётов
└── README.md         # Описание репозитория
```

Отчёты хранятся в формате PDF, исходные тексты отчётов — в формате `.tex`, а вычислительная часть лабораторных работ — преимущественно в Jupyter Notebook.

## Статус проекта

Учебный проект в текущем состоянии содержит выполненные лабораторные работы №1–9, РК №1–2 и НИРС. Материалы предназначены для просмотра результатов, изучения реализации и воспроизведения экспериментов в соответствующем окружении.

## Материалы курса

Общие [методические материалы и задания курса](https://github.com/ugapanyuk/courses_current/wiki/COURSE_TMO_SPRING_IU5_2026/).
