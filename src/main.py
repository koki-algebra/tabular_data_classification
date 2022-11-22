import lightgbm as lgb
from sklearn.metrics import auc, roc_curve

from datasets import UCIIncomeDataset
from transforms import label_encoding

# UCI Income Dataset
train_data = UCIIncomeDataset(is_train=True, transform=label_encoding)
test_data = UCIIncomeDataset(is_train=False, transform=label_encoding)

X_train = train_data.X
y_train = train_data.y
X_test = test_data.X
y_test = test_data.y

# LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "force_row_wise": True
}

# model training
model = lgb.train(
    params=params,
    train_set=lgb_train,
    valid_sets=lgb_test,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(
        stopping_rounds=10,
        verbose=True
    )]
)

# save model
model.save_model("lightgbm.txt")

# prediction
bst = lgb.Booster(model_file="lightgbm.txt")
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = auc(fpr, tpr)
print(f"AUC: {auc}")
