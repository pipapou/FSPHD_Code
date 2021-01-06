import numpy as np
import tensorflow as tf

def lstm(rep, r_split):

    print("Begin LSTM")

    # import variables explicatives X
    X_train = np.load('./Features_' + rep + '_split' + str(r_split) + '/lstm_x_train.npy')
    X_test = np.load('./Features_' + rep + '_split' + str(r_split) + '/lstm_x_test.npy')

    # import variable réponse Y train et test
    y_train = np.load('./Features_' + rep + '_split' + str(r_split) + '/y_train.npy')
    y_test = np.load('./Features_' + rep + '_split' + str(r_split) + '/y_test.npy')

    # import weights W
    w_train = np.load('./Features_' + rep + '_split' + str(r_split) + '/w_train.npy')
    w_test = np.load('./Features_' + rep + '_split' + str(r_split) + '/w_test.npy')

    # paramètres du LSTM
    tf.compat.v1.reset_default_graph()
    hm_epochs = 1000
    batch_size = 250
    nb_inputs = 5
    timesteps = 14
    nb_hidden = 64
    num_layers = 2
    x = tf.compat.v1.placeholder('float', [None, timesteps,nb_inputs],name="X")
    y = tf.compat.v1.placeholder('float',name="Y")
    w = tf.compat.v1.placeholder('float',name="W")

    # random samples and labels of size `bat_size`
    def next_batch(bat_size, data, labels, weights):
        tup_x=[]
        tup_y=[]
        tup_w=[]
        id = np.arange(0 , len(data))
        np.random.shuffle(id)
        for i in range (int(len(data)/bat_size)):
            idx = id[bat_size*i:bat_size*(i+1)]
            tup_x.append([data[j] for j in idx])
            tup_y.append([labels[j] for j in idx])
            tup_w.append([weights[j] for j in idx])
        return np.asarray(tup_x).astype(float), np.asarray(tup_y), np.asarray(tup_w)

    # lstm architecture
    def make_cell():
        Cell=[]
        for _ in range(num_layers):
            c = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nb_hidden, state_is_tuple=True)
            Cell.append(c)
        return Cell

    # lstm prediction
    def recurrent_neural_network(x):
        layer = {'weights': tf.Variable(tf.random.normal([nb_hidden, 1])),
                 'biases': tf.Variable(tf.random.normal([1]))}

        x = tf.unstack(x, timesteps, 1)

        cell = make_cell()
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)
        outputs, states = tf.compat.v1.nn.static_rnn(cell, x, dtype=tf.float32)
        pred = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
        return pred, outputs

    # entrainement
    def train_neural_network(x):
        #init. fonctions
        prediction, features = recurrent_neural_network(x)
        cost = tf.reduce_sum(tf.multiply(tf.square(y - prediction), w))
        optimizer = tf.compat.v1.train.FtrlOptimizer(0.5).minimize(cost)
        total_error = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
        unexplained_error = tf.reduce_sum(tf.square(y - prediction))
        R_squared = 1 - tf.math.divide(unexplained_error, total_error)
        best_test_loss = 10000000000
        best_test_loss_R2 = 0
        best_ep = 0
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                bat=next_batch(batch_size, X_train, y_train, w_train)
                for i in range(int(len(X_train)/batch_size)):
                    epoch_x = bat[0][i]
                    epoch_x = epoch_x.reshape((batch_size, timesteps, nb_inputs))
                    epoch_y = bat[1][i]
                    epoch_w = bat[2][i]
                    i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, w: epoch_w})
                    epoch_loss += c

                train_R2 = R_squared.eval({x: X_train.reshape((-1, timesteps, nb_inputs)), y: y_train})
                test_R2 = R_squared.eval({x: X_test.reshape((-1, timesteps, nb_inputs)), y: y_test})
                test_loss = cost.eval({x: X_test.reshape((-1, timesteps, nb_inputs)), y: y_test, w: w_test})
                print('Epoch', epoch, 'of',hm_epochs,'train_loss:',epoch_loss,'test_loss:',test_loss,'train_R^2:',train_R2,
                     'test_R^2:',test_R2)
                if (test_R2 > best_test_loss_R2) & (test_R2 < train_R2):
                    save_path = saver.save(sess, "./Models/lstm_epa")
                    best_test_loss = test_loss
                    best_test_loss_R2 = test_R2
                    best_ep = epoch

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            # Restore variables from disk.
            model_saver = tf.compat.v1.train.import_meta_graph("./Models/lstm_epa.meta")
            model_saver.restore(sess, "./Models/lstm_epa")
            print("Best model restored")
            graph = tf.compat.v1.get_default_graph()
            x = graph.get_tensor_by_name("X:0")
            features = graph.get_tensor_by_name("rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_41:0")
            predictions = graph.get_tensor_by_name("add:0")

            featTrain = sess.run(features, feed_dict={x: X_train.reshape((-1, timesteps, nb_inputs))})
            featTrain = np.asarray(featTrain, dtype=np.float32)
            np.save('./Features_' + rep + '_split' + str(r_split) + '/lstm_feat_x_train', featTrain)
            featTest = sess.run(features, feed_dict={x: X_test.reshape((-1, timesteps, nb_inputs))})
            featTest = np.asarray(featTest, dtype=np.float32)
            np.save('./Features_' + rep + '_split' + str(r_split) + '/lstm_feat_x_test', featTest)

            PredTrain = sess.run(predictions, feed_dict={x: X_train.reshape((-1, timesteps, nb_inputs))})
            PredTrain = np.asarray(PredTrain, dtype=np.float32)
            np.save('./Features_' + rep + '_split' + str(r_split) + '/lstm_pred_train', PredTrain)
            PredTest = sess.run(predictions, feed_dict={x: X_test.reshape((-1, timesteps, nb_inputs))})
            PredTest = np.asarray(PredTest, dtype=np.float32)
            np.save('./Features_' + rep + '_split' + str(r_split) + '/lstm_pred_test', PredTest)

            print("Features saved in folder \"Features\"")

            print("Test R2 associate with best loss: ", best_test_loss_R2, " reached at epoch: ", best_ep)

        print("End LSTM")

    train_neural_network(x)
