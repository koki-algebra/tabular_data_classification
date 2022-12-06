import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

from utils import get_dataset

dataset, cat_idxs, cat_dims = get_dataset("./data/uci_income/adult.csv", target="salary")

X_train, y_train = dataset["train_labeled"]
X_test, y_test = dataset["test"]

# LightGBM dataset
lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_test = lgb.Dataset(data=X_test, label=y_test, reference=lgb_train)

# parameters
params = {
    "objective": "binary",
    "metric": "binary_error",
    "force_row_wise": True
}

# model training
model = lgb.train(
    params=params,
    train_set=lgb_train,
    valid_sets=lgb_test,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(
        stopping_rounds=100,
        verbose=True
    )]
)

# save model
model.save_model("lightgbm.txt")

# prediction
bst = lgb.Booster(model_file="lightgbm.txt")
y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)

# thresholding
y_pred = np.where(y_pred_prob > 0.5, 1, 0)

# accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"accuracy: {accuracy}")
