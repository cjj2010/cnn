import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
import random
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, LSTM, Bidirectional, Masking
import time

# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(
        name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - \
        result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
        df_op_cond['setting_2'].astype(str) + '_' + \
        df_op_cond['setting_3'].astype(str)

    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']
                   == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
            df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr')[sensors].apply(
        lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit_nr')['unit_nr'].transform(
        create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    data_gen = (list(gen_train_data(df[df['unit_nr'] == unit_nr], sequence_length, columns))
                for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array


def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    label_gen = [gen_labels(df[df['unit_nr'] == unit_nr], sequence_length, label)
                 for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array


def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(
            columns)), fill_value=mask_value)  # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def get_data(dataset, sensors, sequence_length, alpha, threshold):
    # files
    dir_path = './data/'
    train_file = 'train_'+dataset+'.txt'
    test_file = 'test_'+dataset+'.txt'
    # columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names
    # data readout
    train = pd.read_csv((dir_path+train_file), sep=r'\s+', header=None,
                        names=col_names)
    test = pd.read_csv((dir_path+test_file), sep=r'\s+', header=None,
                       names=col_names)
    y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None,
                         names=['RemainingUsefulLife'])

    # create RUL values according to the piece-wise target function
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)

    # remove unused sensors
    drop_sensors = [
        element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(
        X_train_pre, X_test_pre, sensors)

    # exponential smoothing
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80,
                            random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units
    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()):
        # gss returns indexes and index starts at 1
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

        x_train = gen_data_wrapper(
            X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(
            X_train_pre, sequence_length, ['RUL'], train_unit)

        x_val = gen_data_wrapper(
            X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(
            X_train_pre, sequence_length, ['RUL'], val_unit)

    # create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']
# ---------------------------------------------------------------------------------------------------

# --------------------------------------- TRAINING CALLBACKS ----------------------------------------
class save_latent_space_viz(Callback):
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target

    def on_train_begin(self, logs={}):
        self.best_val_loss = 100000

    def on_epoch_end(self, epoch, logs=None):
        encoder = self.model.layers[0]
        if logs.get('val_loss') < self.best_val_loss:
            self.best_val_loss = logs.get('val_loss')
            # viz_latent_space(encoder, self.data, self.target, epoch, False, False)


def get_callbacks(model, data, target):
    model_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30),
        ModelCheckpoint(filepath='./checkpoints/checkpoint', monitor='val_loss',
                        mode='min', verbose=0, save_best_only=True, save_weights_only=True),
        # TensorBoard(log_dir='./logs'),
        save_latent_space_viz(model, data, target)        
    ]
    return model_callbacks


def viz_latent_space(encoder, data, targets=[], epoch='Final', save=False, show=False):
    z, _, _ = encoder.predict(data)
    plt.figure(figsize=(8, 10))
    if len(targets) > 0:
        plt.scatter(z[:, 0], z[:, 1], c=targets)
    else:
        plt.scatter(z[:, 0], z[:, 1])
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    if show:
        plt.show()
    if save:
        plt.savefig('./images/latent_space_epoch'+str(epoch)+'.png')
    return z
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------- MODEL DEFINITION -----------------------------------------
seed = 99
random.seed(seed)
tf.random.set_seed(seed)

def create_model(timesteps, input_dim, intermediate_dim, batch_size, latent_dim, epochs, optimizer):
    # Setup the network parameters:
    timesteps = timesteps
    input_dim = input_dim
    intermediate_dim = intermediate_dim
    batch_size = batch_size
    latent_dim = latent_dim
    epochs = epochs
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    else:
        print("unimplemented optimizer")
        exit(-1)
    masking_value = -99.

    class Sampling(keras.layers.Layer):
        """Uses (z_mean, sigma) to sample z, the vector encoding an engine trajetory."""
        def call(self, inputs):
            mu, sigma = inputs
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return mu + tf.exp(0.5 * sigma) * epsilon

    # ----------------------- Encoder -----------------------
    inputs = Input(shape=(timesteps, input_dim,), name='encoder_input')

    mask = Masking(mask_value=masking_value)(inputs)

    # LSTM encoding
    h = Bidirectional(LSTM(intermediate_dim))(mask)

    # VAE Z layer
    mu = Dense(latent_dim)(h)
    sigma = Dense(latent_dim)(h)

    z = Sampling()([mu, sigma])

    # Instantiate the encoder model:
    encoder = keras.Model(inputs, [z, mu, sigma], name='encoder')
    # print(encoder.summary())
    # -------------------------------------------------------

    # ---------------------- Regressor ----------------------
    reg_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_reg')
    reg_intermediate = Dense(200, activation='tanh')(reg_latent_inputs)
    reg_outputs = Dense(1, name='reg_output')(reg_intermediate)
    # Instantiate the classifier model:
    regressor = keras.Model(reg_latent_inputs, reg_outputs, name='regressor')
    # print(regressor.summary())
    # -------------------------------------------------------

    # -------------------- Wrapper model --------------------
    class RVE(keras.Model):
        def __init__(self, encoder, regressor, decoder=None, **kwargs):
            super(RVE, self).__init__(**kwargs)
            self.encoder = encoder
            self.regressor = regressor
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
            self.decoder = decoder
            if self.decoder != None:
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        @property
        def metrics(self):
            if self.decoder != None:
                return [
                    self.total_loss_tracker,
                    self.kl_loss_tracker,
                    self.reg_loss_tracker,
                    self.reconstruction_loss_tracker
                ]
            else:
                return [
                    self.total_loss_tracker,
                    self.kl_loss_tracker,
                    self.reg_loss_tracker,
                ]

        def train_step(self, data):
            x, target_x = data
            with tf.GradientTape() as tape:
                # kl loss
                z, mu, sigma = self.encoder(x)
                kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                # Regressor
                reg_prediction = self.regressor(z)
                reg_loss = tf.reduce_mean(keras.losses.mse(target_x, reg_prediction))
                # Reconstruction
                if self.decoder != None:
                    reconstruction = self.decoder(z)
                    reconstruction_loss = tf.reduce_mean(keras.losses.mse(x, reconstruction))
                    total_loss = kl_loss + reg_loss + reconstruction_loss
                    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                else:
                    total_loss = kl_loss + reg_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.reg_loss_tracker.update_state(reg_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "reg_loss": self.reg_loss_tracker.result(),
            }

        def test_step(self, data):
            x, target_x = data

            # kl loss
            z, mu, sigma = self.encoder(x)
            kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # Regressor
            reg_prediction = self.regressor(z)
            reg_loss = tf.reduce_mean(keras.losses.mse(target_x, reg_prediction))
            # Reconstruction
            if self.decoder != None:
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(keras.losses.mse(x, reconstruction))
                total_loss = kl_loss + reg_loss + reconstruction_loss
            else:
                total_loss = kl_loss + reg_loss

            return {
                "loss": total_loss,
                "kl_loss": kl_loss,
                "reg_loss": reg_loss,
            }
    # -------------------------------------------------------

    rve = RVE(encoder, regressor)
    rve.compile(optimizer=optimizer)

    return rve
# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # -------------------------------- DATA ---------------------------------
    dataset = input("Enter dataset (FD001, FD002, FD003, FD004): ")

    # sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
    if dataset in ['FD001', 'FD003']:
        sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
    else:
        sensors = ['s_3', 's_4', 's_7', 's_10', 's_11', 's_12']

    sequence_length = 30
    alpha = 0.1
    threshold = 125

    x_train, y_train, x_val, y_val, x_test, y_test = get_data(dataset, sensors, sequence_length, alpha, threshold)
    # -----------------------------------------------------------------------

    # -------------------------------- MODEL --------------------------------
    timesteps = x_train.shape[1]  # l=30
    input_dim = x_train.shape[2]  # n=5
    intermediate_dim = 300
    batch_size = 128
    latent_dim = 2
    epochs = 100  # 10000
    optimizer = 'adam'

    RVE = create_model(timesteps, input_dim, intermediate_dim, batch_size, latent_dim, epochs, optimizer)

    # Callbacks for training
    model_callbacks = get_callbacks(RVE, x_train, y_train)
    # -----------------------------------------------------------------------

    # ------------------------------ TRAINING -------------------------------
    start_time = time.time()
    history = RVE.fit(x_train, y_train, shuffle=True, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), callbacks=model_callbacks, verbose=1)
    end_time = time.time()
    print("Training time: {:.2f} seconds".format(end_time - start_time))
    # -----------------------------------------------------------------------

    # ----------------------------- EVALUATION ------------------------------
    def evaluate(y_true, y_hat, label='test'):
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_hat)
        print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

    train_z = viz_latent_space(RVE.encoder, np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
    test_z = viz_latent_space(RVE.encoder, x_test, y_test.clip(upper=threshold))
    y_hat_train = RVE.regressor.predict(train_z)
    y_hat_test = RVE.regressor.predict(test_z)

    evaluate(np.concatenate((y_train, y_val)), y_hat_train, 'train')
    evaluate(y_test.clip(upper=threshold), y_hat_test, 'test')
    # -----------------------------------------------------------------------
