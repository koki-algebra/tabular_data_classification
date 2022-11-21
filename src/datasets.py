import pandas as pd

class UCIIncomeDataset:
	def __init__(self, is_train: bool) -> None:
		target_column = "salary"

		if is_train:
			train_data = pd.read_csv("../data/uci_income/adult_train.csv")
			self.__X = train_data.drop(target_column, axis=1)
			self.__y = train_data[target_column]

		else:
			test_data = pd.read_csv("../data/uci_income/adult_test.csv")
			self.__X = test_data.drop(target_column, axis=1)
			self.__y = test_data[target_column]

		# categorical columns
		categorical_columns = self.__X.select_dtypes(include="object").columns
		# one-hot encoding
		self.__X = pd.get_dummies(self.__X, columns=categorical_columns, drop_first=True, dummy_na=True)

	@property
	def X(self) -> pd.DataFrame:
		return self.__X

	@X.setter
	def X(self, X):
		self.__X = X

	@property
	def y(self) -> pd.Series:
		return self.__y

	@y.setter
	def y(self, y):
		self.__y = y
