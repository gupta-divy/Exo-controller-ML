from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv1D, BatchNormalization, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.regularizers import l2

def gen_lstm_model(features=54, sequence_length=50):
    # Define model architecture
    model = Sequential([
    LSTM(50, return_sequences=True,input_shape=(None,features)),
    LSTM(50, return_sequences=False),
    Dense(1, activation=None)])
    
    # model = Sequential([
    #     LSTM(128, return_sequences=True,input_shape=(None,features)),
    #     LSTM(64, return_sequences=True, regularizer=l2),
    #     LSTM(32, return_sequences=False),
    #     Dense(32, activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(1, activation=None)])

    #compile the model
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    return model

def gen_FCNN_model(features=54):
    model = Sequential([
        Input(shape=(features,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation=None)])
    #compile the model
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    return model


def gen_tcn_model(features, num_conv_layers=5, num_filters=50, kernel_size=4):
    filters = [num_filters]*num_conv_layers
    dilations=[]
    for i in range(num_conv_layers):
        dilations.append(kernel_size**i)
    effective_window = kernel_size**num_conv_layers
    print("Input Sequence Length should be greater than or equal to ",effective_window)
    inputs = Input(shape=(None, features))

    # stack of 1D convolutional layers with increasing dilation rates
    x = inputs
    eff_hist = 0
    for i in range(num_conv_layers):
        dilation_rate = dilations[i]
        num_channels = filters[i]
        # res_block = tf.keras.layers.LayerNormalization()(x)
        res_block = Conv1D(filters=num_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
        if x.shape[2]!=res_block.shape[2]:
            res_conn = tf.keras.layers.Conv1D(filters=res_block.shape[2], kernel_size=1, activation=None)(x)
        else:
            res_conn = x
        # eff_hist = (kernel_size-1)*dilation_rate
        # res_conn = tf.keras.layers.Cropping1D(cropping=(eff_hist, 0))(res_conn)
        x = tf.keras.activations.relu(res_block+res_conn)
    x = x[:,effective_window-1,:]
    outputs = Dense(1, activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    return model, effective_window



def gen_checkpoint(checkpoint_filepath = 'Model_weights/best_weights.h5'):
    checkpoint = ModelCheckpoint(checkpoint_filepath,monitor='val_mse',verbose=1,save_best_only=True,mode='min')
    return checkpoint


if __name__ == "__main__":
    # Construct model
    model, effective_window = gen_tcn_model(features=19, num_conv_layers=4,  num_filters=50, kernel_size=4)
    print(model.summary)
