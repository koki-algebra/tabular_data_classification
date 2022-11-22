import lightgbm as lgb
from sklearn import metrics

from datasets import UCIIncomeDataset

# UCI Income Dataset
train_data = UCIIncomeDataset(is_train=True)
test_data = UCIIncomeDataset(is_train=False)

target_column = "salary"

X_train = train_data.X
y_train = train_data.y
X_test = test_data.X
y_test = test_data.y

print(X_train.head())

# # LightGBM
# lgb_train = lgb.Dataset(X_train, y_train.astype("int"))
# lgb_test = lgb.Dataset(X_test, y_test.astype("int"), reference=lgb_train)

# params = {"objective": "binary", "metric": "auc"}

# # model training
# model = lgb.train(
#     params=params,
#     train_set=lgb_train,
#     valid_sets=lgb_test,
#     num_boost_round=1000
# )

# # save model
# model.save_model("lightgbm.txt")

# # prediction
# bst = lgb.Booster(model_file="lightgbm.txt")
# y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# # AUC
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
# auc = metrics.auc(fpr, tpr)
# print(auc)
