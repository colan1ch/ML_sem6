
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Настройка конфигурации страницы
st.set_page_config(page_title="Прогнозирование стоимости медстраховки", layout="wide", page_icon="🏥")

# Добавление пользовательских стилей CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #4a4a4a;
        margin-bottom: 5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ff4b4b;
    }
    .prediction-card {
        background-color: #e8f0fe;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #a1c4fd;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #0f9d58;
        margin-top: 10px;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Панель управления и прогнозирования стоимости медицинской страховки")
st.markdown("""
<meta name="description" content="Интерактивная панель для обучения моделей регрессии и прогнозирования годовой стоимости медицинской страховки на основе данных о пациенте.">
""", unsafe_allow_html=True)

# Загрузка и предобработка данных
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('medical_insurance_cost_dataset.csv')
    df_processed = df.copy()
    if 'customer_id' in df_processed.columns:
        df_processed = df_processed.drop('customer_id', axis=1)
        
    categorical_cols = ['gender', 'smoker', 'exercise_level', 'insurance_plan', 'region', 'occupation']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
        
    # Создание новых признаков
    df_processed['age_bmi_interaction'] = df_processed['age'] * df_processed['bmi']
    df_processed['hospitalizations_doctor_visits'] = df_processed['hospitalizations_last_year'] * df_processed['doctor_visits_per_year']
    df_processed['income_age_ratio'] = df_processed['annual_income_usd'] / (df_processed['age'] + 1)
    df_processed['chronic_hospitalization'] = df_processed['chronic_diseases'] * df_processed['hospitalizations_last_year']
    
    X = df_processed.drop('annual_medical_cost_usd', axis=1)
    y = df_processed['annual_medical_cost_usd']
    
    return df, X, y, encoders

# Загрузка данных
df, X, y, encoders = load_and_preprocess_data()

# Инициализация состояния сессии
if 'model' not in st.session_state:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Попытка загрузить предварительно обученную модель
    if os.path.exists('rf_model.pkl') and os.path.exists('scaler.pkl'):
        try:
            with open('rf_model.pkl', 'rb') as f:
                st.session_state.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                st.session_state.scaler = pickle.load(f)
            st.session_state.model_name = "Предварительно обученный случайный лес (с диска)"
        except Exception as e:
            st.session_state.model = None
            st.session_state.scaler = None
            
    if 'model' not in st.session_state or st.session_state.model is None:
        # Обучение модели по умолчанию
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.model_name = "Стандартный случайный лес (n_estimators=100)"

if 'train_metrics' not in st.session_state:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = st.session_state.scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = st.session_state.model
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    st.session_state.train_metrics = {
        "MAE": mean_absolute_error(y_train, y_train_pred),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "R2": r2_score(y_train, y_train_pred)
    }
    st.session_state.test_metrics = {
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "R2": r2_score(y_test, y_test_pred)
    }
    st.session_state.y_test = y_test
    st.session_state.y_test_pred = y_test_pred
    st.session_state.X_columns = X.columns

# Создание вкладок
tab1, tab2 = st.tabs(["📊 Обучение и оценка модели", "🔮 Интерактивный прогноз"])

with tab1:
    st.header("🔧 Обучение и настройка регрессионных моделей")
    st.write("Настройте гиперпараметры модели ниже, затем нажмите **Обучить модель**, чтобы выполнить подгонку и оценить качество на тестовой выборке.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Конфигурация модели")
        model_type = st.selectbox("Выберите семейство моделей", ["Случайный лес (Random Forest)", "Линейная регрессия"])
        
        if model_type == "Случайный лес (Random Forest)":
            n_estimators = st.slider("Количество деревьев (n_estimators)", 10, 300, 100, step=10)
            max_depth = st.slider("Макс. глубина (max_depth)", 1, 30, 10, help="Максимальная глубина деревьев. Меньшая глубина снижает переобучение.")
            min_samples_split = st.slider("Мин. образцов для разделения (min_samples_split)", 2, 10, 2)
        else:
            reg_type = st.selectbox("Тип регуляризации", [
                "Без регуляризации (OLS)",
                "L2 регуляризация (Ridge)",
                "L1 регуляризация (Lasso)"
            ])
            if reg_type != "Без регуляризации (OLS)":
                alpha = st.slider("Сила регуляризации (alpha)", 0.01, 100.0, 1.0, step=0.1)
                
        train_button = st.button("🚀 Обучить модель", use_container_width=True)
        
    if train_button:
        with st.spinner("Обучение модели, пожалуйста, подождите..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if model_type == "Случайный лес (Random Forest)":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                st.session_state.model_name = f"Случайный лес (n_estimators={n_estimators}, max_depth={max_depth})"
            else:
                if reg_type == "Без регуляризации (OLS)":
                    model = LinearRegression()
                    st.session_state.model_name = "Линейная регрессия (OLS)"
                elif reg_type == "L2 регуляризация (Ridge)":
                    model = Ridge(alpha=alpha)
                    st.session_state.model_name = f"Гребневая регрессия Ridge (alpha={alpha})"
                else:
                    model = Lasso(alpha=alpha)
                    st.session_state.model_name = f"Лассо регрессия Lasso (alpha={alpha})"
            
            model.fit(X_train_scaled, y_train)
            
            # Прогноз
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Метрики
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Сохранение в состоянии сессии
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.train_metrics = {"MAE": train_mae, "RMSE": train_rmse, "R2": train_r2}
            st.session_state.test_metrics = {"MAE": test_mae, "RMSE": test_rmse, "R2": test_r2}
            st.session_state.y_test = y_test
            st.session_state.y_test_pred = y_test_pred
            st.session_state.X_columns = X.columns
            
            st.success("Модель успешно обучена!")
            
    with col2:
        st.subheader(f"Текущая модель: {st.session_state.model_name}")
        
        # Отображение метрик
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE (тестовая)</div>
                <div class="metric-value">${st.session_state.test_metrics['MAE']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">RMSE (тестовая)</div>
                <div class="metric-value">${st.session_state.test_metrics['RMSE']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R² (тестовая)</div>
                <div class="metric-value">{st.session_state.test_metrics['R2']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Сравнение с тренировочными метриками
        st.caption(f"Обучающие метрики: MAE: ${st.session_state.train_metrics['MAE']:.2f} | RMSE: ${st.session_state.train_metrics['RMSE']:.2f} | R²: {st.session_state.train_metrics['R2']:.4f}")
        
        # Графики
        st.subheader("Визуализация модели")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Фактические vs Предсказанные
        sns.scatterplot(x=st.session_state.y_test, y=st.session_state.y_test_pred, ax=axes[0], alpha=0.6)
        # Линия диагонали
        min_val = min(st.session_state.y_test.min(), st.session_state.y_test_pred.min())
        max_val = max(st.session_state.y_test.max(), st.session_state.y_test_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], '--r', lw=2)
        axes[0].set_xlabel("Фактическая стоимость (USD)")
        axes[0].set_ylabel("Предсказанная стоимость (USD)")
        axes[0].set_title("Фактические vs. Предсказанные значения")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        # График 2: Важность признаков или Коэффициенты
        model = st.session_state.model
        features = list(st.session_state.X_columns)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
            feat_imp.tail(10).plot(kind='barh', ax=axes[1])
            axes[1].set_title("Топ-10 важности признаков")
            axes[1].set_xlabel("Относительная важность")
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            feat_coef = pd.Series(coefs, index=features).sort_values(ascending=True)
            feat_coef.plot(kind='barh', ax=axes[1], color=np.where(feat_coef >= 0, '#2ecc71', '#e74c3c'))
            axes[1].set_title("Веса признаков")
            axes[1].set_xlabel("Вес коэффициента")
            
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.header("🔮 Оценка стоимости страховки для пациента")
    st.write("Настройте параметры ниже, чтобы описать демографические данные, состояние здоровья и медицинскую историю пациента.")
    
    # Используем форму для группового ввода
    with st.form("patient_input_form"):
        # Макет колонок
        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            st.subheader("📋 Демография")
            age = st.slider("Возраст", 18, 100, 35)
            gender = st.selectbox("Пол", encoders['gender'].classes_)
            region = st.selectbox("Регион", encoders['region'].classes_)
            occupation = st.selectbox("Профессия", encoders['occupation'].classes_)
            annual_income = st.number_input("Годовой доход (USD)", min_value=0, max_value=500000, value=50000, step=1000)
            
        with p_col2:
            st.subheader("💪 Здоровье и привычки")
            bmi = st.slider("Индекс массы тела (ИМТ)", 10.0, 60.0, 25.0, step=0.1)
            smoker = st.selectbox("Статус курения", encoders['smoker'].classes_)
            exercise_level = st.selectbox("Уровень физической активности", encoders['exercise_level'].classes_)
            alcohol_consumption = st.slider("Потребление алкоголя (доз/неделю)", 0, 50, 5)
            chronic_diseases = st.slider("Количество хронических заболеваний", 0, 10, 0)
            
        with p_col3:
            st.subheader("🏥 План и мед. история")
            doctor_visits = st.slider("Визиты к врачу в год", 0, 30, 4)
            hospitalizations = st.slider("Госпитализации в год", 0, 10, 0)
            children = st.slider("Количество детей", 0, 10, 1)
            insurance_plan = st.selectbox("Класс страхового плана", encoders['insurance_plan'].classes_)
            
        submit_prediction = st.form_submit_button("Рассчитать стоимость")
        
    if submit_prediction:
        # Предобработка входных данных
        gender_coded = encoders['gender'].transform([gender])[0]
        smoker_coded = encoders['smoker'].transform([smoker])[0]
        exercise_coded = encoders['exercise_level'].transform([exercise_level])[0]
        insurance_coded = encoders['insurance_plan'].transform([insurance_plan])[0]
        region_coded = encoders['region'].transform([region])[0]
        occupation_coded = encoders['occupation'].transform([occupation])[0]
        
        patient_dict = {
            'age': age,
            'gender': gender_coded,
            'bmi': bmi,
            'children': children,
            'smoker': smoker_coded,
            'region': region_coded,
            'occupation': occupation_coded,
            'annual_income_usd': annual_income,
            'exercise_level': exercise_coded,
            'chronic_diseases': chronic_diseases,
            'doctor_visits_per_year': doctor_visits,
            'hospitalizations_last_year': hospitalizations,
            'alcohol_consumption_per_week': alcohol_consumption,
            'insurance_plan': insurance_coded,
            'age_bmi_interaction': age * bmi,
            'hospitalizations_doctor_visits': hospitalizations * doctor_visits,
            'income_age_ratio': annual_income / (age + 1),
            'chronic_hospitalization': chronic_diseases * hospitalizations
        }
        
        # Создание DataFrame и обеспечение порядка столбцов
        patient_df = pd.DataFrame([patient_dict])
        patient_df = patient_df[st.session_state.X_columns]
        
        # Масштабирование
        patient_scaled = st.session_state.scaler.transform(patient_df)
        
        # Прогноз
        predicted_cost = st.session_state.model.predict(patient_scaled)[0]
        
        # Отображение карточки с прогнозом
        st.markdown(f"""
        <div class="prediction-card">
            <h3>Прогнозируемая годовая стоимость медицинской страховки</h3>
            <div class="prediction-value">${predicted_cost:,.2f}</div>
            <p style="margin-top:15px; color:#6c757d; font-size:0.9rem;">
                Расчет выполнен на основе модели: <strong>{st.session_state.model_name}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
