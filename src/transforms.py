import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoding(X: pd.DataFrame) -> pd.DataFrame:
    for column in X.columns:
        l_encoder = LabelEncoder()
        l_encoder.fit(X[column])
        l_encoded_column = l_encoder.transform(X[column])
        X[column] = pd.Series(l_encoded_column).astype("category")

    return X
