import pandas as pd
from typing import Optional, Callable
from sklearn.preprocessing import LabelEncoder

class UCIIncomeDataset:
    def __init__(self, is_train: bool = True, transform: Optional[Callable] = None) -> None:
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

        # transform
        self.transform = transform

        # label encoding
        l_encoder = LabelEncoder()
        encoded_labels = l_encoder.fit_transform(self.__y)
        self.__y = pd.Series(encoded_labels, name=target_column)

    @property
    def X(self) -> pd.DataFrame:
        if self.transform is not None:
            self.__X = self.transform(self.__X)
        return self.__X

    @property
    def y(self) -> pd.Series:
        return self.__y
