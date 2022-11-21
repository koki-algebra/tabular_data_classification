import pandas as pd
import lightgbm as lgb

from datasets import UCIIncomeDataset

# UCI Income Dataset
train_data = UCIIncomeDataset(is_train=True)
test_data = UCIIncomeDataset(is_train=False)

target_column = "salary"

X_train = train_data.X
y_train = train_data.y
X_test = test_data.X
y_test = test_data.y

# LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)