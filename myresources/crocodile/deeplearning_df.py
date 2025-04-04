
"""
This module contains classes that can be used to preprocess dataframes for deep learning
"""


import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler, RobustScaler  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # type: ignore
from crocodile.file_management import P
# from crocodile.core_modules.core_1 import install_n_import
from typing import Optional, Any, Union


class CategoricalClipper:
    """Use pre onehot or ordinal encoding to create a new category "Other" for values that are less than a certain threshold
Note: this class does not perform imputing. I.e. Null values are ignored at fit, and passed as is at transform. Same behaviour happens with new values at transform time.
This is in contrast to behaviour of sklearn's OrdinalEncoder and OneHotEncoder, which `can` perform missing values handling if instructed to do so.
"""
    def __init__(self, thresh: float = 1.0, others_name: str = 'ClipperOther'):
        self.thresh = thresh
        self.others_name = others_name
        self.columns: Optional[list[str]] = None
        self.pre_percentage_counts: dict[str, 'pd.Series[float]'] = {}
        self.post_percentage_counts: dict[str, 'pd.Series[float]'] = {}
        self.mapper: dict[str, Any] = {}

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]) -> None: self.__dict__ = state

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'CategoricalClipper':
        if verbose:
            print("\n")
            print("Fitting Categorical Clipper".center(100, '-'))

        self.columns = list(df.columns)
        for col in self.columns:
            series = df[col]
            series_na = series.isna()
            if series_na.sum() != 0:
                if verbose:
                    print(f"Column `{col}` has {series.isna().sum()} NaN, NA, None values. These will be dropped before clipping.")
                    print("Percentage of Nulls before dropping = ", series_na.mean() * 100)
                series = series[~series_na]
            self.pre_percentage_counts[col] = series.value_counts(normalize=True) * 100
            self.post_percentage_counts[col], self.mapper[col] = self.create_others_category(self.pre_percentage_counts[col], thresh=self.thresh, others_name=self.others_name)
            name = self.pre_percentage_counts[col].name
            self.pre_percentage_counts[col].name = "Percentage"
            self.pre_percentage_counts[col].index.name = name
            self.post_percentage_counts[col].name = "Percentage"
            self.post_percentage_counts[col].index.name = name
            if verbose: print(f"`{col}` categories pre-clipper:\n{self.pre_percentage_counts[col].to_markdown()}\n\n`{col}` categories post-clipper:\n{self.post_percentage_counts[col].to_markdown()}")
        return self
    def transform(self, df: pd.DataFrame, inplace: bool = True):
        if self.columns is None: raise RuntimeError("Fit the encoder first")
        if not inplace: raise NotImplementedError
        else:
            for col in self.columns:
                series = df.loc[:, col]
                df.loc[:, col] = series.map(self.mapper[col]).copy()
            return df

    def fit_transform(self, df: pd.DataFrame, inplace: bool = True, verbose: bool = True):
        self.fit(df, verbose=verbose)
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
    """
    Note: simliarly to CategoricalClipper, this class does not perform imputing. I.e. Null values are ignored at fit, and passed as is at transform.
    This behaviour is inherited from numpy's ignoring behaviour of NaN when max, min, quantile, etc. are calculated.
    """
    def __init__(self, quant_min: float, quant_max: float):
        self.columns: Optional[list[str]] = None
        self.quant_min: float = quant_min
        self.quant_max: float = quant_max
        self.value_min: dict[str, float] = {}
        self.value_max: dict[str, float] = {}

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]) -> None: self.__dict__ = state

    def fit(self, df: 'pd.DataFrame', verbose: bool = True):
        self.columns = list(df.columns)
        for col in self.columns:
            series = df[col]
            self.value_min[col] = float(series.quantile(self.quant_min))
            self.value_max[col] = float(series.quantile(self.quant_max))
            # without applying float(x), quantile returns np.float64 object which can throw up the dtype at transform time.
        if verbose: self.viz()
        return self

    def viz(self) -> None:
        print("\n")
        print("Clipping Columns".center(100, '-'))
        print(pd.DataFrame([self.value_min, self.value_max], index=["min", "max"]).T)
        print(f"\nQuantiles used for clipping: \n- min: {self.quant_min}\n- max: {self.quant_max}")
        print("-" * 100)
        print("\n")

    def transform(self, df: pd.DataFrame):
        if not self.columns: raise RuntimeError("Fit the encoder first")
        for col in self.columns:
            series = df.loc[:, col]
            df.loc[:, col] = series.clip(lower=self.value_min[col], upper=self.value_max[col])
        return df
    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
    @staticmethod
    def clip_values(y: 'pd.Series[float]', quant_min: float = 0.98, quant_max: float = 0.02):
        los_clip_uppper_hrs = y.quantile(quant_max)
        los_clip_lower_hrs = y.quantile(quant_min)
        return y.clip(lower=los_clip_lower_hrs, upper=los_clip_uppper_hrs)


class DataFrameHandler:
    def __init__(self, scaler: Union[RobustScaler, StandardScaler], imputer: SimpleImputer,
                 cols_ordinal: list[str],
                 cols_categorical: list[str],
                 cols_numerical: list[str],
                 encoder_categorical: OneHotEncoder,
                 encoder_ordinal: OrdinalEncoder,
                 clipper_categorical: CategoricalClipper,
                 clipper_numerical: NumericalClipper
                 ) -> None:
        """
        Use like this:
        ```python

        from crocodile.deeplearning_df import CategoricalClipper, NumericalClipper, DataFrameHandler
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
        from sklearn.impute import SimpleImputer  # type: ignore

        enc = DataFrameHandler(
            scaler=StandardScaler(),
            imputer=SimpleImputer(),
            encoder_categorical=OneHotEncoder(),
            encoder_ordinal=OrdinalEncoder(),
            clipper_categorical=CategoricalClipper(),
            clipper_numerical=NumericalClipper(quant_min=0.02, quant_max=0.98),
            cols_ordinal = [],
            cols_categorical = ['DischargeDisposition', 'pathwayReduced', 'FirstLegDayOfWeek'],
            cols_numerical = ["LOSgenmedLeg", "LOSelse", "FirstLegHourOfDay", "GenMed2ElseLosRatio"],
        )
        df = df_pathways.sample(10_000)
        enc.fit(df)
        df = enc.clip_encode_impute_scale(df)
        data = df[enc.cols_encoded]  # in case df had more columns than declared.

        ```
        """
        self.clipper_categorical: CategoricalClipper = clipper_categorical
        self.clipper_numerical: NumericalClipper = clipper_numerical
        # ordinal data doesn't need to be clipped, because there is no concept of "Other" in ordinal data and there is no cost to having a large number of categories.

        self.cols_numerical: list[str] = cols_numerical
        self.cols_categorical: list[str] = cols_categorical
        self.cols_ordinal: list[str] = cols_ordinal

        self.encoder_ordinal: OrdinalEncoder = encoder_ordinal
        self.encoder_onehot: OneHotEncoder = encoder_categorical

        self.imputer: SimpleImputer = imputer
        self.scaler: Union[RobustScaler, StandardScaler] = scaler

    @property
    def cols_encoded(self):
        """all numerical columns to be used as inputs to the model. Used in getstate, setstate, design model, etc."""
        onehot_names: list[str] = list(self.encoder_onehot.get_feature_names_out())
        return onehot_names + self.cols_ordinal + self.cols_numerical

    def __getstate__(self):
        atts: list[str] = ["scaler", "imputer", "cols_numerical", "cols_ordinal", "cols_onehot", "encoder_onehot", "encoder_ordinal", "clipper_categorical", "clipper_numerical"]
        res = {}
        for att in atts:
            if hasattr(self, att):
                res[att] = getattr(self, att)
        return res

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts the dataframe to numerical format. Missing values are encoded as `pd.NA`, otherwise, encoders will fail to handle them."""
        df.loc[:, self.cols_ordinal] = self.encoder_ordinal.transform(df[self.cols_ordinal])
        tmp = self.encoder_onehot.transform(df[self.cols_categorical])
        if isinstance(tmp, np.ndarray):
            pass
        else:
            tmp = tmp.todense()
        df = df.drop(columns=self.cols_categorical)  # consider inplace=True but make sure it doesn't raise copy warning
        df.loc[:, self.encoder_onehot.get_feature_names_out()] = tmp  # type: ignore
        df.loc[:, self.cols_numerical] = df.loc[:, self.cols_numerical].to_numpy()
        return df

    def impute_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna(np.nan, inplace=True)  # SKlearn Imputer only works with Numpy's np.nan, as opposed to Pandas' pd.NA
        # all_columns = df.columns
        res = self.imputer.transform(df[self.imputer.get_feature_names_out()])
        assert isinstance(res, np.ndarray), f"Imputer returned {type(res)}, but expected np.ndarray"
        df[self.imputer.get_feature_names_out()] = res
        res = self.scaler.transform(df[self.scaler.get_feature_names_out()])
        assert isinstance(res, np.ndarray), f"Scaler returned {type(res)}, but expected np.ndarray"
        df.loc[:, self.scaler.get_feature_names_out()] = res
        # return pd.DataFrame(res, columns=columns)
        return df

    def fit(self, df: 'pd.DataFrame', verbose: bool = True):
        sub_df = df.loc[:, self.cols_categorical]
        self.clipper_categorical.fit(df=sub_df, verbose=verbose)
        df_clipped = self.clipper_categorical.transform(sub_df)
        # because at transform time, the clipping will be applied first and will mutate the data. We don't want to surprise the encoder with new values (e.g. `OtherClipper`)
        self.encoder_onehot.fit(df_clipped[self.cols_categorical])

        self.encoder_ordinal.fit(df[self.cols_ordinal])

        self.clipper_numerical.fit(df.loc[:, self.cols_numerical], verbose=verbose)
        self.imputer.fit(df[self.cols_numerical])
        self.scaler.fit(df[self.cols_ordinal + self.cols_numerical])

    def clip_encode_impute_scale(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        df = self.clipper_categorical.transform(df)
        df = self.encode(df)
        df = self.clipper_numerical.transform(df)
        df = self.impute_standardize(df=df)
        return df


def profile_dataframe(df: pd.DataFrame, save_path: Optional[P] = None):
    # path = data.hp.save_dir.joinpath(data.subpath, f"pandas_profile_report{appendix}.html").create(parents_only=True)
    # from import ProfileReport  # also try pandasgui  # import statement is kept inside the function due to collission with matplotlib
    # report = profile_report(df, title="Pandas Profiling Report", explorative=explorative, silent=silent)
    tmp_path = P.tmpfile(suffix=".parquet")
    df.to_parquet(tmp_path)
    from crocodile.core import run_in_isolated_ve
    if save_path is None:
        save_path = P.tmpfile(suffix=".html")
    pyscript = f"""

from ydata_profiling import ProfileReport
import pandas as pd
df = pd.read_parquet(r'{tmp_path}')
report = ProfileReport(df, title="Profiling Report")
report.to_file(r'{save_path}')

"""
    _launch_script = run_in_isolated_ve(packages=["pandas", "fastparquet", "pyarrow", "ydata-profiling"], pyscript=pyscript)
    print(f"Profile report saved at {save_path}")
    save_path()
    return save_path


def check_for_nan(ip: 'npt.NDArray[Any]') -> int:
    assert len(ip.shape) == 2, f"Expected 2D array, but got {len(ip.shape)}D array"
    total_nan_count: int = 0
    for col_idx in range(ip.shape[1]):
        nan_count = np.isnan(ip[:, col_idx]).sum()
        nan_indices = np.argwhere(np.isnan(ip[:, col_idx]))
        total_nan_count += nan_count
        if nan_count > 0:
            print(f"Column {col_idx}-{ip.shape[1]} has {nan_count} NaNs")
            print(f"Locations of NaNs: {nan_indices[:5]}")
    return total_nan_count


def try_dfh():
    df = pd.DataFrame(np.random.randn(100, 3), columns=list('ABC'))
    dfh = DataFrameHandler(scaler=RobustScaler(), imputer=SimpleImputer(), cols_ordinal=['A'], cols_categorical=['B'], cols_numerical=['C'], encoder_categorical=OneHotEncoder(), encoder_ordinal=OrdinalEncoder(),
                          clipper_categorical=CategoricalClipper(), clipper_numerical=NumericalClipper(quant_min=0.02, quant_max=0.98))
    dfh.fit(df)
    df = dfh.clip_encode_impute_scale(df)


if __name__ == "__main__":
    pass
