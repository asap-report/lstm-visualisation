import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM, Input, Lambda
from keras.models import Model
from keras.regularizers import l1_l2
from keras.utils import np_utils
from matplotlib import cm
from mpl_toolkits import mplot3d
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from tensorflow import set_random_seed


def adjust_input_to_fixed_length(fixed_length, array):
    for i in range(len(array)):
        tmp = np.zeros((fixed_length, 22))
        tmp[-array[i].shape[0]:] = array[i]
        array[i] = tmp
    return array


def load_prepare_data(data_dir, train_size, seed, maxlen=90):
    dirlist = glob.glob(data_dir + '/tctodd?')

    X = []
    Y = []
    lengths = []

    for dir_ in sorted(dirlist):
        for file_ in os.listdir(dir_):
            class_ = file_.split('.')[0][:-2]
            if class_ not in ('his_hers', 'her'):
                array = np.genfromtxt(dir_ + '/' + file_, delimiter='\t')
                if array.shape[0] < maxlen:
                    lengths.append(array.shape[0])
                    Y.append(class_)
                    X.append(array)

    np.random.seed(seed)

    max_length = maxlen

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                                                    train_size=train_size,
                                                                    stratify=Y,
                                                                    random_state=seed)
    X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation,
                                                                  Y_validation,
                                                                  train_size=0.5,
                                                                  random_state=seed)

    X_train = adjust_input_to_fixed_length(max_length, X_train)
    X_validation = adjust_input_to_fixed_length(max_length, X_validation)
    X_test = adjust_input_to_fixed_length(max_length, X_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_validation = np.array(X_validation)

    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y_train = np_utils.to_categorical(le.transform(Y_train))
    Y_test = np_utils.to_categorical(le.transform(Y_test))
    Y_validation = np_utils.to_categorical(le.transform(Y_validation))

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test, le


def prepare_model(n_lstm, metric, X_train, Y_train, kern_reg=(0.01, 0.01),
                  rec_reg=(0.01, 0.01)):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_layer = LSTM(n_lstm, return_sequences=True,
                      kernel_regularizer=l1_l2(kern_reg[0], kern_reg[1]),
                      recurrent_regularizer=l1_l2(rec_reg[0], rec_reg[1]))(
        input_layer)
    last_step_layer = Lambda(lambda x: x[:, -1, :])(lstm_layer)
    output_layer = Dense(Y_train.shape[1], activation='softmax')(
        last_step_layer)

    model = Model(input=input_layer, output=output_layer)
    lstm_activations = Model(input_layer, lstm_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[metric])
    return model, lstm_activations


def fit_model(model, model_weights_filename, X_train, Y_train, X_validation,
              Y_validation, seed, epochs=1000, patience=50, verbosity=0,
              batch_size=32):
    save = ModelCheckpoint(model_weights_filename, monitor='val_loss',
                           verbose=0, save_best_only=True, mode='auto')
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                          verbose=0, mode='auto')
    callbacks_list = [save, early]
    set_random_seed(seed)
    np.random.seed(seed)
    history = model.fit(X_train, Y_train,
                        validation_data=(X_validation, Y_validation),
                        callbacks=callbacks_list,
                        nb_epoch=epochs, batch_size=batch_size,
                        verbose=verbosity)
    return model, history


def load_model(model, model_weights_filename):
    model.load_weights(model_weights_filename)
    return model


def prepare_autoencoder_input(act, noise, seed, min_=20, max_=60):
    Y_autoencoder = np.array(
        [act[i, j, :] for j in range(min_, max_) for i in
         range(act.shape[0])])
    X_autoencoder = Y_autoencoder + np.random.normal(0, noise,
                                                     Y_autoencoder.shape)
    new_order = np.arange(X_autoencoder.shape[0])
    np.random.seed(seed)
    np.random.shuffle(new_order)
    X_autoencoder = X_autoencoder[new_order]
    Y_autoencoder = Y_autoencoder[new_order]
    return X_autoencoder, Y_autoencoder


def prepare_autoencoder(input_shape, dimensions, dense_1=300, dense_2=50,
                        activation='tanh'):
    auto_in = Input(shape=(input_shape,))
    auto_enc = Dense(dense_1, activation='relu')(auto_in)
    auto_enc = Dense(dense_2, activation='relu')(auto_enc)
    auto_enc = Dense(dimensions, activation=activation)(auto_enc)
    auto_dec = Dense(dense_2, activation='relu')(auto_enc)
    auto_dec = Dense(dense_1, activation='relu')(auto_dec)
    auto_dec = Dense(input_shape, activation='linear')(auto_dec)
    decoded = Model(inputs=auto_in, outputs=auto_dec)
    encoded = Model(inputs=auto_in, outputs=auto_enc)
    decoded.compile(optimizer='adam', loss='mse')
    return decoded, encoded


def fit_autoencoder(decoded, X_autoencoder, Y_autoencoder, weights_file_name,
                    seed, patience=5, verbosity=0, batch_size=128, epochs=1000):
    auto_early = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=patience, verbose=verbosity,
                               mode='auto')
    auto_save = ModelCheckpoint(weights_file_name, monitor='val_loss',
                                verbose=verbosity, save_best_only=True,
                                mode='auto')
    auto_callbacks_list = [auto_early, auto_save]
    set_random_seed(seed)
    np.random.seed(seed)
    history = decoded.fit(X_autoencoder, Y_autoencoder, epochs=epochs,
                          validation_split=0.2,
                          callbacks=auto_callbacks_list, batch_size=batch_size,
                          verbose=verbosity)
    return decoded, history


def plot_2D_series(act, set_labels, labels, encoded, ax, c, cmap):
    for l in set_labels:
        where = np.where(labels == l)[0]
        for i in where:
            enc = encoded.predict(act[i])
            plt.plot(enc[:, 0], enc[:, 1], 'k', alpha=0.1,
                     linewidth=1.0)
            sc = plt.scatter(enc[:, 0], enc[:, 1], c=c, cmap=cmap, s=5,
                             alpha=0.5)
            ax.annotate(l, (enc[-1, 0], enc[-1, 1]))


def log_all_1p(a):
    return np.sign(a) * np.log1p(np.abs(a))


def plot_2D(set_labels, encoded, act, labels, title, filename, x_l=-1.05,
            x_h=1.05, y_l=-1.05, y_h=1.05, figsize=(15, 12), frac_labels=0.5,
            linewidth=1, pointsize=5, fontsize=5, linealpha=0.1, min_steps=0,
            max_steps=90, show=True, maxlen=90, log=False, hand=None,
            colormap=cm.viridis, scatter=True, colorbar_show=True,
            last_label=True):
    cmap = colormap
    c = np.linspace(0, maxlen, maxlen)
    fig, ax = plt.subplots(figsize=figsize)

    for l in set_labels:
        where = np.where(labels == l)[0]
        for i in where:
            enc = encoded.predict(act[i])
            if log:
                enc = log_all_1p(enc)
            if hand is None:
                linecolor = 'k'
            elif hand[i]:
                linecolor = 'c'
            elif not hand[i]:
                linecolor = 'm'

            plt.plot(enc[min_steps:max_steps, 0], enc[min_steps:max_steps, 1],
                     linecolor, alpha=linealpha, linewidth=linewidth)
            if scatter:
                sc = plt.scatter(enc[min_steps:max_steps, 0],
                                 enc[min_steps:max_steps, 1],
                                 c=c[min_steps:max_steps], cmap=cmap,
                                 s=pointsize,
                                 alpha=0.5)
            if np.random.uniform() < frac_labels:
                ax.annotate(l, (enc[-1, 0], enc[-1, 1]), fontsize=fontsize)
        if last_label:
            ax.annotate(l, (enc[-1, 0], enc[-1, 1]), fontsize=fontsize)

    if colorbar_show:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=maxlen)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ticks=range(0, maxlen, 20))

    plt.title(title, fontsize=20)
    plt.xlim(x_l, x_h)
    plt.ylim(y_l, y_h)
    plt.rc('axes', labelsize=16)
    fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close("all")


def plot_3D(set_labels, encoded, act, labels, title, filename, x_l=-1.05,
            x_h=1.05, y_l=-1.05, y_h=1.05, figsize=(15, 12), pointsize=5,
            max_steps=136, show=True, log=False):
    cmap = cm.viridis
    c = np.linspace(0, 90, 90)
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')

    for l in set_labels:
        where = np.where(labels == l)[0][:]
        for i in where:
            enc = encoded.predict(act[i])
            if log:
                enc = log_all_1p(enc)
            x = enc[:max_steps, 0],
            y = enc[:max_steps, 1]
            z = enc[:max_steps, 2]
            ax.scatter3D(x, y, z, c=c[:max_steps], cmap=cmap, alpha=0.5,
                         s=pointsize)

    plt.title(title, fontsize=20)
    plt.xlim(x_l, x_h)
    plt.ylim(y_l, y_h)
    ax.set_zlim(y_l, y_h)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close("all")


def plot_2D_mislabeled(set_labels, encoded, act, labels, title, filename,
                       mislabel, pairs, x_l=-1.05, x_h=1.05, y_l=-1.05,
                       y_h=1.05, figsize=(15, 12), frac_labels=0.5, linewidth=1,
                       pointsize=5, fontsize=5, linealpha=0.1, max_steps=90,
                       show=True, maxlen=90, log=False, hand=None,
                       colormap=cm.viridis, scatter=True, last_label=True):
    cmap = colormap
    c = np.linspace(0, maxlen, maxlen)
    fig, ax = plt.subplots(figsize=figsize)

    for l in set_labels:
        where = np.where(labels == l)[0]
        for i in where:
            enc = encoded.predict(act[i])
            if log:
                enc = log_all_1p(enc)
            if hand is None:
                linecolor = 'k'
            elif hand[i]:
                linecolor = 'c'
            elif not hand[i]:
                linecolor = 'm'

            plt.plot(enc[:max_steps, 0], enc[:max_steps, 1], linecolor,
                     alpha=linealpha, linewidth=linewidth)
            if scatter:
                sc = plt.scatter(enc[:max_steps, 0], enc[:max_steps, 1],
                                 c=c[:max_steps], cmap=cmap, s=pointsize,
                                 alpha=0.5)
            if mislabel[i]:
                ax.annotate(l, (enc[-1, 0], enc[-1, 1] + 0.05),
                            fontsize=fontsize + 2, color='blue')
                ax.annotate(pairs[i, 0], (enc[-1, 0], enc[-1, 1] - 0.05),
                            fontsize=fontsize + 2, color='red')
            elif np.random.uniform() < frac_labels:
                ax.annotate(l, (enc[-1, 0], enc[-1, 1]), fontsize=fontsize)
        if last_label:
            ax.annotate(l, (enc[-1, 0], enc[-1, 1]), fontsize=fontsize)

    if scatter:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=maxlen)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ticks=range(0, maxlen, 20))

    plt.title(title, fontsize=20)
    plt.xlim(x_l, x_h)
    plt.ylim(y_l, y_h)
    fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close("all")
