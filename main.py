import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------------
# Кастомные классы и функции (должны быть определены до загрузки модели)
# ------------------------------------------------------------------

class DynamicOHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ohe = None
        self.cat_cols_ = None
        
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.cat_cols_ = X.select_dtypes(include='object').columns.tolist()
        self.ohe = ColumnTransformer(
            transformers=[
                ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.cat_cols_),
            ],
            verbose_feature_names_out=False,
            remainder='passthrough'
        )
        self.ohe.fit(X)
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return self.ohe.transform(X)

        
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col):
        self.group_col = group_col
        self.target_col = target_col
        self.medians_ = None

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group_col)[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for group, median in self.medians_.items():
            mask = (X[self.group_col] == group) & (X[self.target_col].isna())
            X.loc[mask, self.target_col] = median
        if X[self.target_col].isna().any():
            global_median = X[self.target_col].median()
            X[self.target_col].fillna(global_median, inplace=True)
        return X


class Ordinal_mapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping
        self.unknown_value = -1

    def fit(self, X, y=None):
        for col in self.mapping:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in data")
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.mapping.items():
            X[col] = X[col].map(mapping).fillna(self.unknown_value).astype(int)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mas_vnr_mode_ = None

    def fit(self, X, y=None):
        df = X.copy()
        non_na = df[df['MasVnrType'] != 'NA']['MasVnrType']
        if not non_na.empty:
            self.mas_vnr_mode_ = non_na.mode()[0]
        else:
            self.mas_vnr_mode_ = 'NA'
        return self

    def transform(self, X):
        df = X.copy()
        mask1 = (df['MasVnrType'] == 'NA') & (df['MasVnrArea'] > 0)
        df.loc[mask1, 'MasVnrType'] = self.mas_vnr_mode_
        mask2 = (df['MasVnrType'] != 'NA') & (df['MasVnrArea'] == 0)
        df.loc[mask2, 'MasVnrType'] = 'NA'

        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalPorchSF"] = (
            df["OpenPorchSF"] + df["EnclosedPorch"]
            + df["ScreenPorch"] + df["WoodDeckSF"]
        )
        df["TotalBath"] = (
            df["FullBath"] + df["BsmtFullBath"]
            + 0.5 * df["HalfBath"] + 0.5 * df["BsmtHalfBath"]
        )
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)
        df["QualSF"] = df["OverallQual"] * df["GrLivArea"]
        df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
        return df


# ---------- Функции для FunctionTransformer ----------
# Список колонок, которые логарифмировались при обучении (типичный для Ames)
skewed = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
]

def log_transform(X):
    X = X.copy()
    for col in skewed:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X


def bool_to_int(X):
    X = X.copy()
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X


# ------------------------------------------------------------------
# Основная часть Streamlit-приложения
# ------------------------------------------------------------------

# Устанавливаем вывод трансформеров в pandas (удобно для отладки)
sklearn.set_config(transform_output="pandas")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load("pipeline.pkl")

pipeline = load_model()

# Ожидаемые колонки (все признаки, используемые при обучении)
EXPECTED_COLUMNS = list(pipeline.feature_names_in_)

# Боковая панель навигации (теперь только одна страница)
st.sidebar.title("Навигация")
page = st.sidebar.radio("", ["Загрузить CSV"])

# Инициализация состояния для информации о модели
if "show_info" not in st.session_state:
    st.session_state.show_info = False

if st.sidebar.button("Информация о модели"):
    st.session_state.show_info = not st.session_state.show_info

if st.session_state.show_info:
    st.sidebar.subheader("Ожидаемые колонки")
    st.sidebar.write(", ".join(EXPECTED_COLUMNS))
    st.sidebar.subheader("Типы колонок")
    st.sidebar.write("Модель содержит предобработку, включающую заполнение пропусков и кодирование категорий.")
    st.sidebar.write("Для корректной работы загружаемый CSV-файл должен содержать все указанные колонки.")

    # Создаём пример CSV
    example_data = {
        "Id": 1,
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotFrontage": 70,
        "LotArea": 10000,
        "Street": "Pave",
        "Alley": "NA",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 6,
        "OverallCond": 5,
        "YearBuilt": 2000,
        "YearRemodAdd": 2005,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinSF1": 0,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 0,
        "TotalBsmtSF": 0,
        "Heating": "GasA",
        "HeatingQC": "TA",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 1000,
        "2ndFlrSF": 0,
        "LowQualFinSF": 0,
        "GrLivArea": 1000,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FullBath": 1,
        "HalfBath": 0,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 5,
        "Functional": "Typ",
        "Fireplaces": 0,
        "FireplaceQu": "NA",
        "GarageType": "Attchd",
        "GarageYrBlt": 2000,
        "GarageFinish": "Unf",
        "GarageCars": 1,
        "GarageArea": 200,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "P",
        "WoodDeckSF": 0,
        "OpenPorchSF": 0,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": "NA",
        "Fence": "NA",
        "MiscFeature": "NA",
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2010,
        "SaleType": "WD",
        "SaleCondition": "Normal"
    }
    example_df = pd.DataFrame([example_data])
    csv_example = example_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Скачать пример CSV",
        data=csv_example,
        file_name="example_ames.csv",
        mime="text/csv"
    )

# ==================== Страница "Загрузить CSV" ====================
if page == "Загрузить CSV":
    st.title("🌳 Предсказание цены дома по загруженному файлу")

    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными о домах 🏠", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Список категориальных колонок (определяем по типу object, но для надёжности зададим явно)
        categorical_cols = [
            'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
        ]
        numeric_cols = [col for col in EXPECTED_COLUMNS if col not in categorical_cols]
        
        # Приводим числовые колонки к float (для избежания ошибок типов)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        st.subheader("Загруженные данные")
        st.dataframe(df.head(10))

        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"Не хватает колонок: {missing}")
        else:
            predictions = pipeline.predict(df[EXPECTED_COLUMNS])
            predictions = np.expm1(predictions)
            df_result = df.copy()
            df_result["Predicted_Price"] = predictions

            st.subheader("Результат предсказания")
            st.dataframe(df_result)

            total = len(predictions)
            min_price = predictions.min()
            max_price = predictions.max()
            mean_price = predictions.mean()

            st.subheader("Статистика предсказанных цен")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Всего объектов", total)
            col2.metric("Минимальная цена", f"${min_price:,.0f}")
            col3.metric("Максимальная цена", f"${max_price:,.0f}")
            col4.metric("Средняя цена", f"${mean_price:,.0f}")

            st.subheader("Распределение предсказанных цен")
            st.bar_chart(pd.DataFrame(predictions, columns=["Predicted_Price"]))

            csv = df_result.to_csv(index=False)
            st.download_button(
                label="Скачать CSV с предсказаниями",
                data=csv,
                file_name="price_predictions.csv",
                mime="text/csv"
            )