import pandas as pd
import numpy as np
from osgeo import gdal
from sklearn.model_selection import train_test_split

def preprocess(rep, r_split):
    print("Begin preprocessings")
    print("response variable: " + rep + "; random split: " + str(r_split))
    data_rep_src = pd.read_excel("../rep/rep_epa_2009-2018.xlsx") # jeu de données réponses
    Vars_lstm = ['rainfall', 'maize', 'smt', 'tmax', 'tmin'] # vars explicatives LSTM
    Vars_conj = ['world_bank', 'meteo', 'pop', 'ndvi'] # vars explicatives conjonctural
    Vars_struc = ['hosp_educ', 'acled', 'quality_soil', 'elevation', 'waterways'] # vars explicatives structural

    ### preprocessing LSTM
    data_rep = data_rep_src

    # import des données explicatives
    for D in Vars_lstm:
        data_temp = pd.read_excel("../data_explicatives_" + rep + "/data_"+ D + ".xlsx") # données explicatives
        data_rep = pd.merge(data_rep, data_temp, how="left", on=['REGION', 'PROVINCE', 'COMMUNE', 'ANNEE'])

    # Agrégation par mois
    data_aggr = data_rep.iloc[0:len(data_rep),0:9]
    for i in range(2): # année t-1 et t
        for j in range(5, 12): # mois de mai (5) à novembre (11)
            if 'ndvi' in Vars_lstm:
                data_aggr['ndvi{0}t-{1}'.format(j, 1 - i)] = data_rep.filter(regex='ndvi_.*{0}\(.*t-{1}'.format(j, 1 - i)).max(axis=1)
            if 'smt' in Vars_lstm:
                data_aggr['smt{0}t-{1}'.format(j,1-i)] = data_rep.filter(regex='smt_.*{0}\(.*t-{1}'.format(j,1-i)).max(axis=1)
            if 'rainfall' in Vars_lstm:
                data_aggr['rainfall{0}t-{1}'.format(j,1-i)] = data_rep.filter(regex='rainfall.*{0}\(.*t-{1}'.format(j,1-i)).sum(axis=1)
            if 'maize' in Vars_lstm:
                data_aggr['maize{0}t-{1}'.format(j,1-i)] = data_rep.filter(regex='{0}maize-.*\(.*t-{1}'.format(j,1-i))
            if 'sorgho' in Vars_lstm:
                data_aggr['sorgho{0}t-{1}'.format(j,1-i)] = data_rep.filter(regex='{0}sorgho-.*\(.*t-{1}'.format(j,1-i))
            if 'tmax' in Vars_lstm:
                data_aggr['tmax{0}t-{1}'.format(j,1-i)] = data_rep.filter(regex='t_max.*{0}\(t-{1}'.format(j,1-i))
            if 'tmin' in Vars_lstm:
                data_aggr['tmin{0}t-{1}'.format(j,1-i)] = data_rep.filter(regex='t_min.*{0}\(t-{1}'.format(j,1-i))

    # normalisation: on centre réduit les vars (X-X.mean)/X.std
    for v in Vars_lstm:
        data_aggr.loc[:, data_aggr.columns.str.startswith(v)]=(data_aggr.loc[:, data_aggr.columns.str.startswith(v)]-
                                                               data_aggr.loc[:, data_aggr.columns.str.startswith(v)].stack().mean())/\
                                                              data_aggr.loc[:, data_aggr.columns.str.startswith(v)].stack().std()

    # on retire lignes avec variable réponse Nan
    data_aggr = data_aggr[pd.notna(data_aggr[rep])].reset_index().drop(['index'], axis=1)

    # stockage données X du LSTM en array numpy
    dataX_lstm = np.array(data_aggr.iloc[0:len(data_aggr),9:data_aggr.shape[1]])

    ### preprocessing vars conj + struc
    data_aggr = data_rep_src

    # import des données
    for D in Vars_conj:
        if D == "bm":
            data_temp = pd.read_excel("../data_explicatives_" + rep + "/data_bm.xlsx")  # données explicatives
            data_aggr = pd.merge(data_aggr, data_temp, how="left", on=['ANNEE'])

        else:
            data_temp = pd.read_excel("../data_explicatives_" + rep + "/data_" + D + ".xlsx") # données explicatives
            data_aggr = pd.merge(data_aggr, data_temp, how="left", on=['REGION', 'PROVINCE', 'COMMUNE',"ANNEE"])

    # import données explicatives
    for D in Vars_struc:
        data_temp = pd.read_excel("../data_explicatives_" + rep + "/data_" + D + ".xlsx") # données explicatives
        data_aggr = pd.merge(data_aggr, data_temp, how="left", on=['REGION', 'PROVINCE', 'COMMUNE'])

    # normalisation: on centre réduit les vars (X-X.mean)/X.std
    for v in data_aggr.iloc[0:len(data_aggr),9:data_aggr.shape[1]].columns:
        data_aggr[v] = (data_aggr[v] - data_aggr[v].mean())/data_aggr[v].std()

    # on retire lignes avec variable réponse Nan
    data_aggr = data_aggr[pd.notna(data_aggr[rep])].reset_index().drop(['index'], axis=1)

    # stockage des données X conj et struc en array numpy
    dataX_CS = np.array(data_aggr.iloc[0:len(data_aggr),9:data_aggr.shape[1]])

    ### stockage en array numpy infos commune et années pour cnn
    dataInfo = []
    for i in range(len(data_aggr)):
        dataInfo.append([data_aggr['ANNEE'][i], data_aggr['ID_COM'][i]])
    dataInfo = np.array(dataInfo)

    ### stockage en array numpy reponse (Y)
    dataY = []
    for i in range(len(data_aggr)):
        dataY.append([data_aggr[rep][i]])
    dataY = np.array(dataY)

    # stockage des weight (W) (pour chaque observation: W = racine(nombre de ménages)/racine(total des ménages)
    data_aggr['Count'] = np.sqrt(data_aggr['Count']) / np.sqrt(data_aggr['Count']).sum()
    dataW = []
    for i in range(len(data_aggr)):
        dataW.append([data_aggr['Count'][i]])
    dataW = np.array(dataW)

    # division train/test
    X_train_lstm, X_test_lstm, X_train_CS, X_test_CS, y_train, y_test, w_train, w_test, info_train, info_test = train_test_split(
        dataX_lstm, dataX_CS, dataY, dataW, dataInfo, test_size=0.15, random_state=r_split)

    ### preprocessing CNN

    # import des infos communes auxquels appartiennent les pixels
    raster_com = gdal.Open('../data_cnn/epa_100m_com.tif')
    raster_com = np.array(raster_com.ReadAsArray())

    drep = dict() # dico des réponses de chaque année
    pop = dict() # dico des populations de chaque année

    cult = gdal.Open('../data_cnn/crop_mean_100m.tif') # import donnees cultures
    forest = gdal.Open('../data_cnn/forest_mean_100m.tif')  # import donnees forets
    built = gdal.Open('../data_cnn/built_mean_100m.tif')  # import donnees zones construites

    cult = np.array(cult.ReadAsArray())
    forest = np.array(forest.ReadAsArray())
    built = np.array(built.ReadAsArray())

    # import des rasters réponse et population de chaque année
    for annee in range(2009, 2019): # import des rasters réponse et population
        drep[annee] = gdal.Open('../data_cnn/' + rep + '_100m/epa_100m_' + rep + '_' + str(annee) + '.tif')
        pop[annee] = gdal.Open('../data_cnn/population_100m/bfa_ppp_' + str(annee) + '.tif')

        #transformation en données réponse et population en array numpy
        drep[annee] = np.array(drep[annee].ReadAsArray())
        pop[annee] = np.array(pop[annee].ReadAsArray())

        #pixels à valeur nulle transformés en nan
        drep[annee][drep[annee] <= 0] = np.nan
        pop[annee][pop[annee] <= 0] = np.nan

        #normalisation pixels population (X-mean/std)
        pop[annee] = pop[annee] - np.nanmean(pop[annee]) / np.nanstd(pop[annee])

    info_pix_cnn = [] # données sur les couples (commune, année) associés à chaque pixel
    dataX_CNN = [] # stockage patchs pixels population
    dataY_CNN = [] # stockage pixels réponse
    longueur = 10 # longueur  des patchs
    pas = 30 # écart entre 2 pixels sélectionnés

    # on rempli info_pix, dataX_CNN, data_Y_CNN
    for annee in range(2009, 2019): # pour chaque année
        for i in range(int(longueur/2), drep[annee].shape[0] - int(longueur/2), pas): # on balaie horizontalement les pixels par pas de 30 avec une marge = int(longueur/2)
            for j in range(int(longueur/2), drep[annee].shape[1] - int(longueur/2), pas): # on balaie verticalement les pixels  par pas de 30 avec une marge = int(longueur/2)
                """
                conditions pour intégrer un pixel réponse + un patch population + 3 patchs oqp_sol au jeu de données du CNN:
                - le pixel réponse ne doit pas être Nan
                - aucun pixel du patch population associé ne doit être Nan
                - tous les pixels du patch population associé doivent appartenir à la même commune que le pixel réponse
                """
                if not ((np.isnan(drep[annee][i, j])) |
                        (np.isnan(pop[annee][i - int(longueur/2):i + int(longueur/2), j - int(longueur/2):j + int(longueur/2)]).any()) |
                        (raster_com[i - int(longueur/2):i + int(longueur/2), j - int(longueur/2):j + int(longueur/2)] != raster_com[i, j]).any()):

                    info_pix_cnn.append([annee, raster_com[i, j], len(info_pix_cnn)])
                    dataX_CNN.append([pop[annee][i - int(longueur/2):i + int(longueur/2), j - int(longueur/2):j + int(longueur/2)],
                                      cult[i - int(longueur/2):i + int(longueur/2), j - int(longueur/2):j + int(longueur/2)],
                                      forest[i - int(longueur/2):i + int(longueur/2), j - int(longueur/2):j + int(longueur/2)],
                                      built[i - int(longueur/2):i + int(longueur/2), j - int(longueur/2):j + int(longueur/2)]])
                    dataY_CNN.append([drep[annee][i, j]])

    # on transforme les données en array numpy

    dataX_CNN = np.array(dataX_CNN)
    dataY_CNN = np.array(dataY_CNN)
    info_pix_cnn = np.array(info_pix_cnn, dtype=int)

    # transformation des données infos en dataframe pour le merge entre les infos pixels et les infos (commune, année) train / test
    info_pix_cnn = pd.DataFrame(info_pix_cnn, columns=['ANNEE', 'CODE_COM', 'line'])
    info_train = pd.DataFrame(info_train, columns=['ANNEE', 'CODE_COM'])
    info_test = pd.DataFrame(info_test, columns=['ANNEE', 'CODE_COM'])

    # merge infos pixels et infos (commune, année) train et test
    info_pix_cnn_train = pd.merge(info_pix_cnn, info_train, how='inner', on=['ANNEE', 'CODE_COM'])
    info_pix_cnn_test = pd.merge(info_pix_cnn, info_test, how='inner', on=['ANNEE', 'CODE_COM'])

    # merge avec pandas effectué, on retransforme en array numpy
    info_pix_cnn_train = np.array(info_pix_cnn_train)
    info_pix_cnn_test = np.array(info_pix_cnn_test)

    # division des données X et Y en train et test
    X_pix_train_cnn = dataX_CNN[info_pix_cnn_train[:, 2]]
    X_pix_test_cnn = dataX_CNN[info_pix_cnn_test[:, 2]]
    y_pix_train_cnn = dataY_CNN[info_pix_cnn_train[:, 2]]
    y_pix_test_cnn = dataY_CNN[info_pix_cnn_test[:, 2]]

    # plus besoin des infos sur les localisations des pixels, on les retire
    info_pix_cnn_train = np.delete(info_pix_cnn_train, 2, 1)
    info_pix_cnn_test = np.delete(info_pix_cnn_test, 2, 1)

    # save cnn elements
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_info_pix_train.npy", info_pix_cnn_train)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_info_pix_test.npy", info_pix_cnn_test)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_x_pix_train.npy", X_pix_train_cnn)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_x_pix_test.npy", X_pix_test_cnn)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_y_pix_train.npy", y_pix_train_cnn)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_y_pix_test.npy", y_pix_test_cnn)

    # save explicative variables X
    np.save("./Features_" + rep + "_split" + str(r_split) + "/lstm_x_train.npy", X_train_lstm)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/lstm_x_test.npy", X_test_lstm)

    np.save("./Features_" + rep + "_split" + str(r_split) + "/cs_x_train.npy", X_train_CS)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/cs_x_test.npy", X_test_CS)

    # save response Y
    np.save("./Features_" + rep + "_split" + str(r_split) + "/y_train.npy", y_train)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/y_test.npy", y_test)

    # save weights W
    np.save("./Features_" + rep + "_split" + str(r_split) + "/w_train.npy", w_train)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/w_test.npy", w_test)

    # save infos (commune, année) des données train et test
    np.save("./Features_" + rep + "_split" + str(r_split) + "/info_train.npy", info_train)
    np.save("./Features_" + rep + "_split" + str(r_split) + "/info_test.npy", info_test)

    print("Variables and features saved in folder \"features\" ")
    print("End preprocessings")
