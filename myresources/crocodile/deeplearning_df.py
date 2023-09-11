
"""
This module contains classes that can be used to preprocess dataframes for deep learning
"""


from typing import Optional, Any, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import crocodile.toolbox as tb
# from crocodile.deeplearning import DataReader


class CategoricalClipper:
    """Use pre onehot or ordinal encoding to create a new category "Other" for values that are less than a certain threshold"""
    def __init__(self, thresh: float = 1.0, others_name: str = 'Other'):
        self.thresh = thresh
        self.others_name = others_name
        self.columns: Optional[list[str]] = None
        self.pre_percentage_counts: dict[str, 'pd.Series[float]'] = {}
        self.post_percentage_counts: dict[str, 'pd.Series[float]'] = {}
        self.mapper: dict[str, Any] = {}

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]) -> None: self.__dict__ = state

    def fit(self, df: pd.DataFrame) -> 'CategoricalClipper':
        print("\n")
        print(f"Fitting Categorical Clipper".center(100, '-'))

        self.columns = list(df.columns)
        for col in self.columns:
            series = df[col]
            self.pre_percentage_counts[col] = series.value_counts(normalize=True) * 100
            self.post_percentage_counts[col], self.mapper[col] = self.create_others_category(self.pre_percentage_counts[col], thresh=self.thresh, others_name=self.others_name)
            name = self.pre_percentage_counts[col].name
            self.pre_percentage_counts[col].name = "Percentage"
            self.pre_percentage_counts[col].index.name = name
            self.post_percentage_counts[col].name = "Percentage"
            self.post_percentage_counts[col].index.name = name
            print(f"`{col}` categories pre-clipper:\n{self.pre_percentage_counts[col].to_markdown()}\n\n`{col}` categories post-clipper:\n{self.post_percentage_counts[col].to_markdown()}")
        return self
    def transform(self, df: pd.DataFrame, inplace: bool = True):
        if self.columns is None: raise RuntimeError("Fit the encoder first")
        if not inplace: raise NotImplementedError
        else:
            for col in self.columns:
                series = df.loc[:, col]
                df.loc[:, col] = series.map(self.mapper[col])
            return df

    def fit_transform(self, df: pd.DataFrame, inplace: bool = True):
        self.fit(df)
        return self.transform(df, inplace=inplace)

    @staticmethod
    def create_others_category(percentage_counts: 'pd.Series[float]', thresh: float = 1.0, others_name: str = 'Other'):
        orig_caregories = percentage_counts.index.tolist()
        other_cats = percentage_counts[percentage_counts < thresh].index
        mask = percentage_counts.index.isin(other_cats)
        others_percentage = percentage_counts[mask].sum()

        percentage_counts_updated = percentage_counts[~mask]
        percentage_counts_updated[others_name] = others_percentage

        # tmp = series.value_counts(normalize=True) * 100
        # print(f"Percentage of `Other` after collation = {series.value_counts(normalize=True).Other}")
        mapper = {cat: others_name for cat in other_cats}
        mapper.update({cat: cat for cat in orig_caregories if cat not in other_cats})
        return percentage_counts_updated, mapper


class NumericalClipper:
    def __init__(self, quant_min: float, quant_max: float):
        self.columns: Optional[list[str]] = None
        self.quant_min: float = quant_min
        self.quant_max: float = quant_max
        self.value_min: dict[str, float] = {}
        self.value_max: dict[str, float] = {}

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]) -> None: self.__dict__ = state

    def fit(self, df: 'pd.DataFrame'):
        self.columns = list(df.columns)
        for col in self.columns:
            series = df[col]
            self.value_min[col] = series.quantile(self.quant_min)
            self.value_max[col] = series.quantile(self.quant_max)
        self.viz()
        return self

    def viz(self) -> None:
        print("\n")
        print(f"Clipping Columns".center(100, '-'))
        print(pd.DataFrame([self.value_min, self.value_max], index=["min", "max"]).T)
        print(f"\nQuantiles used for clipping: \n- min: {self.quant_min}\n- max: {self.quant_max}")
        print(f"-" * 100)
        print("\n")

    def transform(self, df: pd.DataFrame):
        if not self.columns: raise RuntimeError("Fit the encoder first")
        for col in self.columns:
            series = df.loc[:, col]
            df.loc[:, col] = series.clip(lower=self.value_min[col], upper=self.value_max[col])
        return df

    @staticmethod
    def clip_values(y: 'pd.Series[float]', quant_min: float = 0.98, quant_max: float = 0.02):
        los_clip_uppper_hrs = y.quantile(quant_max)
        los_clip_lower_hrs = y.quantile(quant_min)
        return y.clip(lower=los_clip_lower_hrs, upper=los_clip_uppper_hrs)


class DataFrameHander:
    def __init__(self, scaler: Union[RobustScaler, StandardScaler], imputer: SimpleImputer, cols_ordinal: list[str], cols_onehot: list[str], cols_numerical: list[str],
                 encoder_onehot: OneHotEncoder,
                 encoder_ordinal: OrdinalEncoder,
                 clipper_categorical: CategoricalClipper,
                 clipper_numerical: NumericalClipper
                 ) -> None:
        self.cols_ordinal: list[str] = cols_ordinal
        self.cols_onehot: list[str] = cols_onehot
        self.cols_numerical: list[str] = cols_numerical

        self.scaler: Union[RobustScaler, StandardScaler] = scaler
        self.imputer: SimpleImputer = imputer

        self.encoder_onehot: OneHotEncoder = encoder_onehot
        self.encoder_ordinal: OrdinalEncoder = encoder_ordinal

        self.clipper_categorical: CategoricalClipper = clipper_categorical
        self.clipper_numerical: NumericalClipper = clipper_numerical
    def __getstate__(self):
        atts: list[str] = ["scaler", "imputer", "cols_numerical", "cols_ordinal", "cols_onehot", "encoder_onehot", "encoder_ordinal", "clipper_categorical", "clipper_numerical"]
        res = {}
        for att in atts:
            if hasattr(self, att):
                res[att] = getattr(self, att)
        return res

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, save_path: Optional[tb.P] = None, silent: bool = False, explorative: bool = True):
        # path = data.hp.save_dir.joinpath(data.subpath, f"pandas_profile_report{appendix}.html").create(parents_only=True)
        profile_report = tb.install_n_import(library="ydata_profiling", package="ydata-profiling").ProfileReport
        # from pandas_profiling import ProfileReport
        # profile_report = pandas_profiling.()
        # from import ProfileReport  # also try pandasgui  # import statement is kept inside the function due to collission with matplotlib
        report = profile_report(df, title="Pandas Profiling Report", explorative=explorative)
        if save_path is not None: report.to_file(save_path)

    @staticmethod
    def gui_dataframe(df: 'pd.DataFrame'): tb.install_n_import("pandasgui").show(df)

    def encode(self, df: pd.DataFrame, precision: str) -> pd.DataFrame:
        """Converts the dataframe to numerical format. Missing values are encoded as `pd.NA`, otherwise, encoders will fail to handle them."""
        df[self.cols_ordinal] = self.encoder_ordinal.transform(df[self.cols_ordinal])
        tmp = self.encoder_onehot.transform(df[self.cols_onehot])
        df.drop(columns=self.cols_onehot, inplace=True)
        df[self.encoder_onehot.get_feature_names_out()] = tmp
        df[self.cols_numerical] = df[self.cols_numerical].to_numpy().astype(precision)
        return df

    def impute_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna(np.nan, inplace=True)  # SKlearn Imputer only works with Numpy's np.nan, as opposed to Pandas' pd.NA
        columns = df.columns
        res = self.imputer.transform(df)
        assert isinstance(res, np.ndarray), f"Imputer returned {type(res)}, but expected np.ndarray"
        res = self.scaler.transform(pd.DataFrame(res, columns=columns))
        assert isinstance(res, np.ndarray), f"Scaler returned {type(res)}, but expected np.ndarray"
        return pd.DataFrame(res, columns=columns)


if __name__ == "__main__":
    pass
