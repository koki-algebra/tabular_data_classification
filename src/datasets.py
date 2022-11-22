import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class UCIIncomeDataset:
    def __init__(self, is_train: bool) -> None:
        target_column = "salary"
        df: pd.DataFrame

        # read csv file
        if is_train:
            df = pd.read_csv("../data/uci_income/adult_train.csv")
        else:
            df = pd.read_csv("../data/uci_income/adult_test.csv")

        # set features and label
        self.__X = df.drop(target_column, axis=1)
        self.__y = df[target_column]

        # categorical columns
        categorical_columns = self.__X.select_dtypes(include="object").columns

        # one-hot encoding
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(self.__X[categorical_columns].values.reshape(-1, 1))
        encoded_data = encoder.transform(
            self.__X[categorical_columns].values.reshape(-1, 1)
        )
        self.__X = pd.DataFrame(encoded_data)

        # label encoding
        l_encoder = LabelEncoder()
        l_encoder.fit(self.__y)
        encoded_label = l_encoder.transform(self.__y)
        self.__y = pd.Series(encoded_label)

        print(self.__X.shape)
        print(self.__y.shape)

    @property
    def X(self) -> pd.DataFrame:
        return self.__X

    @property
    def y(self) -> pd.Series:
        return self.__y
