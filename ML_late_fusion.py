import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

def ml_late(rep, r_split):
    print("Begin Machine Learning on predictions")

    # import variable réponse Y train et test
    y_train = np.load("./Features_" + rep + "_split" + str(r_split) + '/y_train.npy')
    y_test = np.load("./Features_" + rep + "_split" + str(r_split) + '/y_test.npy')

    # import prédictions de Y train et test
    # for lstm model
    pred_lstm_train = np.load("./Features_" + rep + "_split" + str(r_split) + '/lstm_pred_train.npy')
    pred_lstm_test = np.load("./Features_" + rep + "_split" + str(r_split) + '/lstm_pred_test.npy')
    #for cnn model
    pred_cnn_train = np.load("./Features_" + rep + "_split" + str(r_split) + '/cnn_pred_train.npy')
    pred_cnn_test = np.load("./Features_" + rep + "_split" + str(r_split) + '/cnn_pred_test.npy')

    ### vars init

    cs_train = np.load("./Features_" + rep + "_split" + str(r_split) + '/cs_x_train.npy')
    cs_test = np.load("./Features_" + rep + "_split" + str(r_split) + '/cs_x_test.npy')

    # import weights
    w_train = np.load("./Features_" + rep + "_split" + str(r_split) + '/w_train.npy')
    w_test = np.load("./Features_" + rep + "_split" + str(r_split) + '/w_test.npy')

    print("separate models responses")

    print(" lstm prediction; Test R2: %f" % r2_score(y_test, pred_lstm_test, sample_weight=w_test))
    print(" cnn prediction; Test R2: %f" % r2_score(y_test, pred_cnn_test, sample_weight=w_test))

    ### RF vars conj + struc
    print("RF vars conj + struc")

    params_cs = {'n_estimators': 900, 'max_depth': 20, 'random_state': 1}
    linreg = RandomForestRegressor(**params_cs)
    linreg.fit(cs_train, y_train.ravel(), sample_weight=w_train.ravel())
    pred_rf_train = linreg.predict(cs_train)
    pred_rf_test = linreg.predict(cs_test)
    print("rf ; conj + struc ; Test R2: %f" % r2_score(y_test, pred_rf_test, sample_weight=w_test))

    ### Regression on responses
    print("responses aggregation")
    # concatenate predictions
    # cnn + lstm + rf
    preds_train_3 = np.concatenate((pred_cnn_train.reshape(-1, 1), pred_lstm_train.reshape(-1, 1),
                                   pred_rf_train.reshape(-1, 1)), axis=1)
    preds_test_3 = np.concatenate((pred_cnn_test.reshape(-1, 1), pred_lstm_test.reshape(-1, 1),
                                  pred_rf_test.reshape(-1, 1)), axis=1)

    alf = 0.2
    linreg = Ridge(alpha=alf)
    linreg.fit(preds_train_3, y_train, sample_weight=w_train.ravel())
    predictions = linreg.predict(preds_test_3)
    score = r2_score(y_test, predictions, sample_weight=w_test)
    print("linear regression on cnn + rf + lstm responses ; ridge alpha = " + str(alf) + "Test R2: " + str(score))

