
"""Test behaviour of pandas DataFrame with respect to missing values behaviour.
The following shows how imputing can be done implicitly by the encoders.
This still doesn't dictate the imputation strategy per se, but it does show the implementation.
For example, one can impute missing values with average values, or with the most frequent value and perform that handling of na values before the encoding.

Preporcessing of data is done in the following steps.
1- Indentify data types: [numerical, ordinal, onehot]
It is crucial to do this first, because the subsequent processing relies on this information.
Numerical data go down this pathway:
A- Impute missing values.
B- Clip the range (if the scalar is robust, this might be superflous).
C- Scale the data.
Categorical data go down this pathway:
A- Both ordinal and onehot data are (optionally imputes first)
B- then encoded (which can still handle the imputation internally).
C- Scale (only for ordinally encoded data)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # type: ignore

data = {
    'category': ['A', 'B', 'A', 'C', pd.NA, 'B', 'C', 'A'],
}

df = pd.DataFrame(data)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # <====== Make sure this is set to 'ignore' to handle missing values.
fit_data = df[['category']].dropna()  # <====== Make sure to drop NA values, so Null values are not thought of as a category.
encoder.fit(fit_data)
transform_data = pd.DataFrame({'category': ['A', 'B', 'D', pd.NA, np.nan, None]})  # notice that .isna() captures all of these
transformed = encoder.transform(transform_data)  # <==== see how D, NA, np.nan, and None are handles as [0, 0, 0, ...]
print("Encoded categories:")
print(encoder.get_feature_names_out(['category']))
print(transformed)


data = {
    'category': ['A', 'B', 'A', 'C', pd.NA, 'B', 'C', 'A'],
}

df = pd.DataFrame(data)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['C', 'A', 'B']])  # <====== Handle unknown values with a special value.
fit_data = df[['category']].dropna()  # <====== Make sure to drop NA values, so Null values are not thought of as a category.
encoder.fit(fit_data)
transform_data = pd.DataFrame({'category': ['A', 'B', 'D', pd.NA, np.nan, None]})  # notice that .isna() captures all of these
transformed = encoder.transform(transform_data)  # <==== see how D, NA, np.nan, and None are handles as [0, 0, 0, ...]
print("Encoded categories:")
print(encoder.get_feature_names_out(['category']))
print(transformed)
