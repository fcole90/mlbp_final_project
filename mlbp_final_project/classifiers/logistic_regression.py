import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


import mlbp_final_project.utils.data_loader as data_loader

def main():
    X = data_loader.load_train_data()
    y = data_loader.load_train_labels()
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    class_weights = np.array([np.sum(y[y == label]) for label in range(1, 11)]) / y.shape[0]
    print("class weights:", class_weights)

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print("X_test shape:", X_test.shape)
        # print("y_test shape", y_test.shape)
        # print("X_train shape", X_train.shape)
        # print("y_train shape", y_train.shape)
        print("\n")
        # Dummy
        dummy_labels = [0] * 10
        dummy_labels[0] = 1
        y_pred_dummy = np.tile([1], y_test.shape[0]).reshape(y_test.shape)
        y_pred_lloss_dummy = np.tile(dummy_labels, y_test.shape[0]).reshape(y_test.shape[0], (len(dummy_labels)))
        print("[{}]".format(i),
              "using dummy",
              "| Acc: %.2f" % accuracy_score(y_test, y_pred_dummy),
              "| logLoss: %.2f" % log_loss(y_true=y_test,
                                   y_pred=y_pred_lloss_dummy,
                                   labels=list(range(1, 11))))

        # Class proportion
        labels_data = class_weights
        y_pred_data = np.tile(np.argmax(labels_data) + 1, y_test.shape[0]).reshape(y_test.shape)
        y_pred_lloss_data = np.tile(labels_data, y_test.shape[0]).reshape(y_test.shape[0], (len(labels_data)))
        print("[{}]".format(i),
              "using data",
              "| Acc: %.2f" % accuracy_score(y_test, y_pred_data),
              "| logLoss: %.2f" % log_loss(y_true=y_test,
                                   y_pred=y_pred_lloss_data,
                                   labels=list(range(1, 11))))


        # Standard Logistic Regression
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        # print("log_reg classes:", log_reg.classes_, type(log_reg.classes_))
        y_pred_lloss_log_reg = log_reg.predict_proba(X_test)
        # print("y_pred shape:", y_pred.shape)
        classes_log_reg = log_reg.classes_
        print("[{}]".format(i),
              "using log_reg",
              "| Acc: %.2f" % log_reg.score(X_test, y_test),
              "| logLoss: %.2f" % log_loss(y_true=y_test,
                                   y_pred=y_pred_lloss_log_reg,
                                   labels=classes_log_reg))

        # LogLoss combined with class proportion
        labels_data = class_weights
        data_multipliers = 1 + labels_data - np.average(labels_data)
        y_pred_lloss_log_comb_weight = y_pred_lloss_log_reg * data_multipliers
        y_pred_acc_log_comb_weight = np.argmax(y_pred_lloss_log_comb_weight, axis=1) + 1
        print("[{}]".format(i),
              "using log_combo_data",
              "| Acc: %.2f" % accuracy_score(y_test, y_pred_acc_log_comb_weight),
              "| logLoss: %.2f" % log_loss(y_true=y_test,
                                           y_pred=y_pred_lloss_log_comb_weight,
                                           labels=list(range(1, 11))))






if __name__ == "__main__":
    main()
