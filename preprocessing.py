import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------- Constants ----------

cols_drop = [
    "Street", "Utilities", "PoolQC", "PoolArea",
    "MiscFeature", "MiscVal", "Alley",
    "RoofMatl", "Heating", "Condition2",
]

text_none = [
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "Fence", "MasVnrType",
]

num_none = [
    "GarageArea", "GarageCars", "GarageYrBlt", "MasVnrArea",
    "BsmtFullBath", "BsmtHalfBath", "TotalBsmtSF", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF",
]

mode_cols = [
    "MSZoning", "Functional", "Exterior1st", "Exterior2nd",
    "Electrical", "KitchenQual", "SaleType",
]

qual_map = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
fin_map = {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
exp_map = {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
grg_map = {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}
func_map = {
    "Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3,
    "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7,
}

ordinal_encoding_columns = {
    "ExterQual": qual_map, "ExterCond": qual_map,
    "HeatingQC": qual_map, "KitchenQual": qual_map,
    "FireplaceQu": qual_map, "GarageQual": qual_map,
    "GarageCond": qual_map, "BsmtQual": qual_map,
    "BsmtCond": qual_map, "BsmtFinType1": fin_map,
    "BsmtFinType2": fin_map, "BsmtExposure": exp_map,
    "GarageFinish": grg_map, "Functional": func_map,
    "LotShape": {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0},
    "LandSlope": {"Gtl": 1, "Mod": 2, "Sev": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}

skewed = [
    "LotArea", "LotFrontage", "GrLivArea", "TotalBsmtSF",
    "1stFlrSF", "WoodDeckSF", "OpenPorchSF", "MasVnrArea",
    "TotalSF", "QualSF", "QualTotalSF", "TotalPorchSF",
]


# ---------- Custom transformers ----------

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


# ---------- Function transformers ----------

def log_transform(X):
    X = X.copy()
    for col in skewed:
        X[col] = np.log1p(X[col])
    return X


def bool_to_int(X):
    X = X.copy()
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X
