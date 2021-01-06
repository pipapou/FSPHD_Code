import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil
from sklearn.metrics import r2_score

def cnn(rep, r_split):

    print("Begin CNN on population and land cover data")

    # import variables
    X_train = np.load("./Features_" + rep + "_split" + str(r_split) + "/cnn_x_pix_train.npy") # patchs population train
    X_test = np.load("./Features_" + rep + "_split" + str(r_split) + "/cnn_x_pix_test.npy") # patchs population test
    y_train = np.load("./Features_" + rep + "_split" + str(r_split) + "/cnn_y_pix_train.npy") # pixels réponse train
    y_test = np.load("./Features_" + rep + "_split" + str(r_split) + "/cnn_y_pix_test.npy") # pixels réponse test
    y_train_com = np.load("./Features_" + rep + "_split" + str(r_split) + "/y_train.npy") # réponses du train au niveau commune
    y_test_com = np.load("./Features_" + rep + "_split" + str(r_split) + "/y_test.npy") # réponse du test au niveau commune

    L = 10  # largeur et longueur des patchs de pixels population
    X_train = X_train.reshape(-1, L, L, 4)
    X_test = X_test.reshape(-1, L, L, 4)


    # paramètres du LSTM
    tf.compat.v1.reset_default_graph()
    hm_epochs = 50
    batch_size = 500
    nbfilter1 = 32 # nombre de filtres de la 1ere couche de convolution
    nbfilter2 = 64 # nombre de filtres de la 2eme couche de convolution
    nbfilter3 = 128 # nombre de filtres de la 3eme couche de convolution
    shapeconv = 3 # dimension des patchs de convolution
    shapepool = 2 # dimension des patchs de pooling
    finalshape = ceil(L/8) # dimension des données à la fin du cnn

    x = tf.compat.v1.placeholder('float', [None, L, L, 4], name="X")
    y = tf.compat.v1.placeholder('float', name="Y")

    # random samples and labels of size `bat_size`
    def next_batch(bat_size, data, labels):
        tup_x = []
        tup_y = []
        id = np.arange(0, len(data))
        np.random.shuffle(id)
        for i in range(int(len(data) / bat_size)):
            idx = id[bat_size * i:bat_size * (i + 1)]
            tup_x.append([data[j] for j in idx])
            tup_y.append([labels[j] for j in idx])
        return np.asarray(tup_x).astype(float), np.asarray(tup_y)

    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=shapepool):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(x):
        weights = {
            'wc1': tf.get_variable('W0', shape=(shapeconv, shapeconv, 4, nbfilter1), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('W1', shape=(shapeconv, shapeconv, nbfilter1, nbfilter2), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable('W2', shape=(shapeconv, shapeconv, nbfilter2, nbfilter3), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W3', shape=(finalshape * finalshape * nbfilter3, nbfilter3), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(nbfilter3, 1), initializer=tf.contrib.layers.xavier_initializer()),
        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(nbfilter1), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(nbfilter2), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(nbfilter3), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(nbfilter3), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),
        }

        # first conv
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])

        # first Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=shapepool)

        # second conv
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

        # second Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=shapepool)

        # third conv
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

        # third Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=shapepool)

        # Fully connected layer
        # Reshape cnn output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        outputs = tf.nn.relu(fc1)
        pred = tf.add(tf.matmul(outputs, weights['out']), biases['out'])

        return pred, outputs

    # entrainement
    def train_neural_network(x):
        # init. fonctions
        prediction, features = conv_net(x)
        cost = tf.reduce_sum(tf.square(y - prediction))
        optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)
        total_error = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
        unexplained_error = tf.reduce_sum(tf.square(y - prediction))
        R_squared = 1 - tf.math.divide(unexplained_error, total_error)
        best_test_loss = 10000000000
        best_test_loss_R2 = 0
        best_ep = 0
        saver = tf.compat.v1.train.Saver()
        # entrainement
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                bat = next_batch(batch_size, X_train, y_train)
                for i in range(int(len(X_train) / batch_size)):
                    epoch_x = bat[0][i]
                    epoch_x = epoch_x
                    epoch_y = bat[1][i]

                    i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                train_R2 = R_squared.eval({x: X_train, y: y_train})
                test_R2 = R_squared.eval({x: X_test, y: y_test})
                test_loss = cost.eval({x: X_test, y: y_test})
                print('Epoch', epoch, 'of', hm_epochs, 'train_loss:', epoch_loss, 'test_loss:', test_loss, 'train_R^2:',
                      train_R2, 'test_R^2:', test_R2)
                if test_R2 > best_test_loss_R2:
                    save_path = saver.save(sess, "./Models/cnn_epa")
                    best_test_loss_R2 = test_R2
                    best_ep = epoch

        # utilisation du best model
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            # init paramètres
            model_saver = tf.compat.v1.train.import_meta_graph("./Models/cnn_epa.meta")
            model_saver.restore(sess, "./Models/cnn_epa")
            print("Best model restored")
            graph = tf.compat.v1.get_default_graph()
            x = graph.get_tensor_by_name("X:0")
            features = graph.get_tensor_by_name("Relu_3:0")
            predictions = graph.get_tensor_by_name("Add_1:0")
            info_pix_train = np.load("./Features_" + rep + "_split" + str(r_split) + "/cnn_info_pix_train.npy")
            info_pix_test = np.load("./Features_" + rep + "_split" + str(r_split) + "/cnn_info_pix_test.npy")
            info_train = pd.DataFrame(np.load("./Features_" + rep + "_split" + str(r_split) + "/info_train.npy"))
            info_test = pd.DataFrame(np.load("./Features_" + rep + "_split" + str(r_split) + "/info_test.npy"))

            # calcul des features et prédictions des pixels
            FeatPixTrain = sess.run(features, feed_dict={x: X_train})
            FeatPixTrain = np.asarray(FeatPixTrain, dtype=np.float32)

            FeatPixTest = sess.run(features, feed_dict={x: X_test})
            FeatPixTest = np.asarray(FeatPixTest, dtype=np.float32)

            PredPixTrain = sess.run(predictions, feed_dict={x: X_train})
            PredPixTrain = np.asarray(PredPixTrain, dtype=np.float32)

            PredPixTest = sess.run(predictions, feed_dict={x: X_test})
            PredPixTest = np.asarray(PredPixTest, dtype=np.float32)

            # merge des features et prédictions avec les infos (commune, année) pour chaque pixel
            FeatPixTrain = pd.DataFrame(np.concatenate((info_pix_train, FeatPixTrain), axis=1))
            FeatPixTest = pd.DataFrame(np.concatenate((info_pix_test, FeatPixTest), axis=1))
            PredPixTrain = pd.DataFrame(np.concatenate((info_pix_train, PredPixTrain), axis=1))
            PredPixTest = pd.DataFrame(np.concatenate((info_pix_test, PredPixTest), axis=1))

            # regrouper les features et prédictions des pixels par (commune, année)
            FeatTrain = FeatPixTrain.groupby([0, 1]).agg({key: "mean" for key in range(2, len(FeatPixTrain.columns))}).reset_index()
            FeatTrain = pd.merge(info_train, FeatTrain, how='left', on=[0, 1])
            FeatTrain = FeatTrain.drop([0, 1], axis=1)
            FeatTrain = np.array(FeatTrain)
            np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_feat_x_train", FeatTrain)

            FeatTest = FeatPixTest.groupby([0, 1]).agg({key: "mean" for key in range(2, len(FeatPixTest.columns))}).reset_index()
            FeatTest = pd.merge(info_test, FeatTest, how='left', on=[0, 1])
            FeatTest = FeatTest.drop([0, 1], axis=1)
            FeatTest = np.array(FeatTest)
            np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_feat_x_test", FeatTest)
            print("Features saved in folder \"Features\"")

            PredTrain = PredPixTrain.groupby([0, 1]).agg({2:"mean"}).reset_index()
            PredTrain = pd.merge(info_train, PredTrain, how='left', on=[0, 1])
            PredTrain = PredTrain.drop([0, 1], axis=1)
            PredTrain = np.array(PredTrain)
            np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_pred_train", PredTrain)

            PredTest = PredPixTest.groupby([0, 1]).agg({2:"mean"}).reset_index()
            PredTest = pd.merge(info_test, PredTest, how='left', on=[0, 1])
            PredTest = PredTest.drop([0, 1], axis=1)
            PredTest = np.array(PredTest)
            np.save("./Features_" + rep + "_split" + str(r_split) + "/cnn_pred_test", PredTest)

            Final_R2 = r2_score(y_test_com, PredTest)

            print("Test R2 associate with best loss: ", Final_R2, " reached at epoch: ", best_ep)

        print("End CNN")

    train_neural_network(x)