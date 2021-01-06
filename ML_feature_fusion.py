import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def ml(rep, r_split):
    print("Begin Machine Learning on initial variables and NN features")

    # import variable réponse Y train et test
    y_train = np.load('./Features_' + rep + '_split' + str(r_split) + '/y_train.npy')
    y_test = np.load('./Features_' + rep + '_split' + str(r_split) + '/y_test.npy')

    # import weights
    w_train = np.load('./Features_' + rep + '_split' + str(r_split) + '/w_train.npy')
    w_test = np.load('./Features_' + rep + '_split' + str(r_split) + '/w_test.npy')

    # import des variables explicatives initiales (x_train, x_test) et features en sortie des NN (feat_x_train, feat_x_test)
    data_list = dict()
    for data in ['lstm', 'cnn']:
        data_list['feat_x_train_{0}'.format(data)] = np.load("./Features_" + rep + "_split" + str(r_split) + "/" + data + "_feat_x_train.npy")
        data_list['feat_x_test_{0}'.format(data)] = np.load("./Features_" + rep + "_split" + str(r_split) + "/" + data + "_feat_x_test.npy")
    for data in ['lstm', 'cs']:
        data_list['x_train_{0}'.format(data)] = np.load("./Features_" + rep + "_split" + str(r_split) + "/" + data + "_x_train.npy")
        data_list['x_test_{0}'.format(data)] = np.load("./Features_" + rep + "_split" + str(r_split) + "/" + data + "_x_test.npy")

    #concaténation des variables initiales
    x_train_tot = np.concatenate((data_list['x_train_lstm'], data_list['x_train_cs']), axis=1)
    x_test_tot = np.concatenate((data_list['x_test_lstm'], data_list['x_test_cs']), axis=1)

    # concaténation des features
    feat_x_train_cnn = data_list['feat_x_train_cnn']
    feat_x_train_lstm = data_list['feat_x_train_lstm']

    feat_x_train_cnn_cs = np.concatenate((feat_x_train_cnn, data_list['x_train_cs']), axis=1)

    feat_x_train_tot = np.concatenate((feat_x_train_cnn, data_list['x_train_cs']), axis=1)
    feat_x_train_tot = np.concatenate((feat_x_train_tot, data_list['feat_x_train_lstm']), axis=1)

    feat_x_test_cnn = data_list['feat_x_test_cnn']
    feat_x_test_lstm = data_list['feat_x_test_lstm']

    feat_x_test_cnn_cs = np.concatenate((feat_x_test_cnn, data_list['x_test_cs']), axis=1)

    feat_x_test_tot = np.concatenate((feat_x_test_cnn, data_list['x_test_cs']), axis=1)
    feat_x_test_tot = np.concatenate((feat_x_test_tot, data_list['feat_x_test_lstm']), axis=1)


    ### RF
    ## sur les variables initiales
    params = {'n_estimators': 900, 'max_depth': 20, 'random_state': 1}

    rf = RandomForestRegressor(**params)
    rf.fit(data_list['x_train_lstm'], y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(data_list['x_test_lstm'])
    print("RF(time series) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    rf = RandomForestRegressor(**params)
    rf.fit(x_train_tot, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(x_test_tot)
    print("RF(init. variables) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    ## sur les features
    # vars lstm seules
    rf = RandomForestRegressor(**params)
    rf.fit(feat_x_train_lstm, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(feat_x_test_lstm)
    print("RF(lstm features) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    # vars cnn seules
    rf = RandomForestRegressor(**params)
    rf.fit(feat_x_train_cnn, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(feat_x_test_cnn)
    print("RF(cnn features) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    # vars cnn + cs
    rf = RandomForestRegressor(**params)
    rf.fit(feat_x_train_cnn_cs, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(feat_x_test_cnn_cs)
    print("RF(cnn features + cs vars) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    # toutes les features
    rf = RandomForestRegressor(**params)
    rf.fit(feat_x_train_tot, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(feat_x_test_tot)
    print("RF(all features) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))
