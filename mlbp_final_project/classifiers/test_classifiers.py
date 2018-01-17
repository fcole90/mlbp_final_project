import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


import mlbp_final_project.utils.data_loader as data_loader


LOG_REG_SOLVER = 'liblinear'  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
MULTICLASS = 'multinomial'  # 'ovr', 'multinomial'


def print_data_line(line, name, acc_val, log_loss_val):
    print("[{}]".format(line),
          "using", name,
          "| Acc: %.5f" % acc_val,
          "| logLoss: %.5f" % log_loss_val)


def log_reg_combo_class(class_weights, predict_proba, y_predict, classes_log_reg):
    """Combines log_reg with the class weights.

    Parameters
    ----------
    class_weights
    predict_proba

    Returns
    -------

    """
    # print(classes_log_reg)
    # print(class_weights)
    # class_weights = class_weights[classes_log_reg.astype(int) - 1]
    # print(class_weights)
    data_multipliers = 1 + class_weights - np.average(class_weights)

    best_predict_combined = np.argmax(predict_proba, axis=1) + 1
    best_predict_proba_combined = predict_proba

    # print(y_predict.shape, best_predict_combined.shape)
    # print(y_predict.shape, best_predict_proba_combined.shape)

    best_acc = accuracy_score(y_predict, best_predict_combined)
    best_lloss = log_loss(y_true=y_predict, y_pred=best_predict_proba_combined, labels=classes_log_reg)

    best_factor_mult_acc = np.nan
    best_factor_norm_acc = np.nan

    best_factor_mult_lloss = np.nan
    best_factor_norm_lloss = np.nan

    for factor_mult in np.arange(0.1, 2.5, 0.1):
        for factor_norm in np.arange(0.9, 1.1, 0.1):

            # print(factor_mult, factor_norm)
            predict_proba_combined = (predict_proba * (1.001 - factor_mult)) * (data_multipliers * factor_mult)
            predict_proba_combined /= np.max(predict_proba_combined, axis=1)[:, None] * factor_norm
            predict_combined = np.argmax(predict_proba_combined, axis=1) + 1

            acc = accuracy_score(y_predict, predict_combined)
            lloss = log_loss(y_true=y_predict, y_pred=predict_proba_combined, labels=classes_log_reg)

            if acc > best_acc:
                best_acc = acc
                best_factor_mult_acc = factor_mult
                best_factor_norm_acc = factor_norm
                best_predict_combined = predict_combined

            if lloss < best_lloss:
                best_lloss = lloss
                best_factor_mult_lloss = factor_mult
                best_factor_norm_lloss = factor_norm
                best_predict_proba_combined = predict_proba_combined

    print("\t\tmult lloss:", best_factor_mult_lloss,
          "- norm lloss:", best_factor_norm_lloss,
          "- mult acc:", best_factor_mult_acc,
          "- norm acc:", best_factor_norm_acc)

    return best_predict_combined, best_predict_proba_combined


def log_reg_sum_class(class_weights, predict_proba):
    """Sum class weights to log_reg probabilities.

    Parameters
    ----------
    class_weights
    predict_proba

    Returns
    -------

    """
    predict_proba_sum = predict_proba
    predict_proba_sum[np.max(predict_proba_sum, axis=1) < 0.3] += (1 + class_weights - np.average(class_weights)) * 2
    predict_sum = np.argmax(predict_proba_sum, axis=1) + 1
    return predict_sum, predict_proba_sum


def run_test_classifiers():
    X = data_loader.load_train_data()
    y = data_loader.load_train_labels()
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    class_weights = np.array([np.sum(y[y == label]) for label in range(1, 11)]) / y.shape[0]
    print("class weights:", class_weights)

    kf = KFold(n_splits=5, shuffle=True)
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
        print_data_line(i, "dummy", accuracy_score(y_test, y_pred_dummy), log_loss(y_true=y_test,
                                                                                   y_pred=y_pred_lloss_dummy,
                                                                                   labels=list(range(1, 11))))

        # Class proportion
        labels_data = class_weights
        predict_class_weights = np.tile(np.argmax(labels_data) + 1, y_test.shape[0]).reshape(y_test.shape)
        predict_proba_class_weights = np.tile(labels_data, y_test.shape[0]).reshape(y_test.shape[0], (len(labels_data)))
        print_data_line(i, "class_weights",
                        accuracy_score(y_test,
                                       predict_class_weights),
                        log_loss(y_true=y_test,
                                 y_pred=predict_proba_class_weights,
                                 labels=list(range(1, 11))))

        # Standard Logistic Regression
        log_reg = LogisticRegression(solver=LOG_REG_SOLVER, n_jobs=12, max_iter=300)
        log_reg.fit(X_train, y_train)
        # print("log_reg classes:", log_reg.classes_, type(log_reg.classes_))
        predict_proba_log_reg = log_reg.predict_proba(X_test)
        # print("y_pred shape:", y_pred.shape)
        classes_log_reg = log_reg.classes_
        print_data_line(i, "log_reg", log_reg.score(X_test, y_test), log_loss(y_true=y_test,
                                                                              y_pred=predict_proba_log_reg,
                                                                              labels=classes_log_reg))

        # LogReg combined with class proportion
        predict_combo, predict_proba_combo = log_reg_combo_class(class_weights,
                                                                 predict_proba_log_reg,
                                                                 y_test,
                                                                 classes_log_reg)
        print_data_line(i, "log_reg_combo_weights",
                        accuracy_score(y_test,
                                       predict_combo),
                        log_loss(y_true=y_test,
                                 y_pred=predict_proba_combo,
                                 labels=classes_log_reg))

        # LogReg summed to class proportion when undecided
        predict_sum, predict_proba_sum = log_reg_sum_class(class_weights, predict_proba_log_reg)
        print_data_line(i, "log_reg_combo_sum",
                        accuracy_score(y_test,
                                       predict_sum),
                        log_loss(y_true=y_test,
                                 y_pred=predict_proba_sum,
                                 labels=classes_log_reg))


    # # Calculate test and save it
    # X_test = data_loader.load_test_data()
    #
    # # Standard Logistic Regression
    # log_reg = LogisticRegression(solver=LOG_REG_SOLVER, n_jobs=12)
    # log_reg.fit(X, y)
    # # print("log_reg classes:", log_reg.classes_, type(log_reg.classes_))
    # predict_proba_log_reg = log_reg.predict_proba(X_test)
    # y_pred_acc_log_reg = log_reg.predict(X_test)
    # data_loader.save_csv_output(predict_proba_log_reg, "log-loss_LogisticRegression_{}_{}".format(LOG_REG_SOLVER))
    # data_loader.save_csv_output(y_pred_acc_log_reg, "accuracy_LogisticRegression_{}_{}".format(LOG_REG_SOLVER))

    # # LogLoss combined with class proportion
    # predict_combo, predict_proba_combo = log_reg_combo_class(class_weights, predict_proba_log_reg, y, log_reg.classes_)
    # data_loader.save_csv_output(predict_combo, "log-loss_LogisticRegressionCombo_{}".format(LOG_REG_SOLVER))
    # data_loader.save_csv_output(predict_proba_combo, "accuracy_LogisticRegressionCombo_{}".format(LOG_REG_SOLVER))
    #
    # # LogReg summed to class proportion when undecided
    # predict_sum, predict_proba_sum = log_reg_sum_class(class_weights, predict_proba_log_reg)
    # data_loader.save_csv_output(predict_combo, "log-loss_LogisticRegressionSum_{}".format(LOG_REG_SOLVER))
    # data_loader.save_csv_output(predict_proba_combo, "accuracy_LogisticRegressionSum_{}".format(LOG_REG_SOLVER))


def test_suite():
    global MULTICLASS
    global LOG_REG_SOLVER

    for lr_solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        for multiclass in ['ovr', 'multinomial']:
            MULTICLASS = multiclass
            LOG_REG_SOLVER = lr_solver
            try:
                print("\n\nSolver: ", lr_solver, "- multiclass:", multiclass)
                run_test_classifiers()
            except Exception as e:
                print("Catched:", str(e))


if __name__ == "__main__":
    test_suite()
    # run_test_classifiers()
