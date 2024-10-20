import numpy as np
import random
import tensorflow as tf
import re
import os

from keras.models import Model
from keras.losses import Loss
from keras import layers
from keras.callbacks import Callback

# Se obtienen 500 arrays y se juntan todos los puntos hasta que diverge del safe set, 
# no hay separadores -1,-1
    #Mean: 2759.181 Median: 2249.5 Standard Deviation: 1922.72 Min: 1431 Max: 32100
# Separamos con -1, -1 cada vez que una orbita escapa y metemos el siguiente punto
    #Mean: 3758.852 Median: 3292.0 Standard Deviation: 1894.60 Min: 2346 Max: 37281

def Extended_Function_Dataset(data, mean_extension, test, b, separated_orbits):
    Us = data['Us'][-1,:]
    xVectors = data['xVectors']

    if test == False:
        Ic = random.randint(0, Us.shape[0]-1)
    else:
        Ic = b

    Us = Us[Ic] #quitarlo para hacer cheating

    xVectors = np.nan_to_num(xVectors, nan=-1) # hay algunos nan de matlab
    xVectors= np.reshape(xVectors, (1000* 50, 2))
    xVectors = xVectors[~np.any(xVectors == -1, axis=1)]

    if separated_orbits == True:
        # We separate each orbit with a [-1, -1] separator
        Separated_xVectors = []
        for i in range(len(xVectors) - 1):
            Separated_xVectors.append(xVectors[i])
            if xVectors[i][1] != xVectors[i + 1][0]:
                Separated_xVectors.append([-1, -1])
        # Append the last row
        Separated_xVectors.append(xVectors[-1])
        # Convert result back to numpy array
        Separated_xVectors = np.array(Separated_xVectors)
        xVectors = Separated_xVectors

    if xVectors.shape[0] > mean_extension:
        xVectors = xVectors[:mean_extension, :]
    elif xVectors.shape[0] < mean_extension:
        num_padding_rows = mean_extension - xVectors.shape[0]
        padding = np.full((num_padding_rows, 2), -1)
        xVectors = np.vstack((xVectors, padding))
    
    return xVectors, Us, Ic

#ME LO HE CARGADO Y VA A HABER QUE ARREGGLARLO PARA FUNCIONES DISTINTAS A LA SMOOTH
def array_filter_smooth(array): #Funcion para filtrar los -1,-1 de los xVectors
    filtered_array = [array[0]]
    # Iterate through the array starting from the second element

    for i in range(len(array)):
        # Check if the current pair contains -1
        if -1 in array[i]:
            # Check if the previous pair is different or if it is the first pair
            if i == 0 or not np.array_equal(array[i - 1], array[i]):
                filtered_array.append(array[i])
        else:
            # If the pair does not contain -1, add it to the filtered list
            filtered_array.append(array[i])

    # Convert the filtered list back to a numpy array
    filtered_array = np.array(filtered_array)
    filtered_array = filtered_array[1:]
    filtered_array = array[~((array == -1).sum(axis=1) == 1)]

    return filtered_array

def smooth_path_generator(mix_maps, fractal_maps, fractal_maps_route, smooth_maps_route, smooth_folders, n_of_files, total_data):
    if mix_maps == True:
        coin_flip = random.randint(0, 1)
        
        if coin_flip == 0:
            random_file = random.randint(1, total_data[0]-1)
            file_route = fractal_maps_route + str(random_file) + '.mat'
        else:
            random_folder = random.choice(smooth_folders)
            random_file = random.randint(1, n_of_files[smooth_folders.index(random_folder)]-1)
            file_route = smooth_maps_route + random_folder + '/maps/' + str(random_file) + '.mat'

    else:
        if fractal_maps == True:
            random_file = random.randint(1, total_data[0]-1)
            file_route = fractal_maps_route + str(random_file) + '.mat'
        else:
            random_folder = random.choice(smooth_folders)
            random_file = random.randint(1, n_of_files[smooth_folders.index(random_folder)]-1)
            file_route = smooth_maps_route + random_folder + '/maps/' + str(random_file) + '.mat'

    return random_file, file_route

def model_details(model, learning_rate, batch_size, loss_function):
    num_layers = len(model.layers)
    layer_units = "-".join([str(layer.output_shape[-1]) for layer in model.layers if hasattr(layer, 'output_shape')])
    reg_type = []
    reg_value = []

    # Assuming we use either L1 or L2 regularization; check kernel_regularizer for each dense layer
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            reg = str(layer.kernel_regularizer).split()[0]
            reg_type.append(reg.split('(')[0])
            reg_value.append(reg.split('=')[-1].replace(')', ''))

    reg_type = "-".join(reg_type) if reg_type else "None"
    reg_value = "-".join(reg_value) if reg_value else "0"

    activations = "-".join([layer.activation.__name__ for layer in model.layers if hasattr(layer, 'activation')])
    dropout_rates = "-".join([str(layer.rate) for layer in model.layers if isinstance(layer, layers.Dropout)])

    optimizer_name = type(model.optimizer).__name__

    model_details = f"{model.name}_layers{num_layers}_units[{layer_units}]_reg[{reg_type}_{reg_value}]_" \
                  f"act[{activations}]_dropout[{dropout_rates}]_lr{learning_rate}_opt{optimizer_name}_" \
                  f"batch{batch_size}_loss{loss_function}"

    return model_details

# Define the learning rate schedule function
def lr_warmup_scheduler(epoch, lr, initial_lr ,warmup_epochs, total_epochs, decay_factor):
    if epoch < warmup_epochs:
        # Linear warmup: increase learning rate
        learning_rate = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Exponential decay: reduce learning rate after warmup
        learning_rate = initial_lr * decay_factor ** (((epoch + 1) - warmup_epochs) / (total_epochs - warmup_epochs))

    return learning_rate

class ArchitectureLogger(Callback):
    def __init__(self, tensorboard_logs):
        super(ArchitectureLogger, self).__init__()
        self.custom_writer = tf.summary.create_file_writer(tensorboard_logs + '/train')
        self.custom_writer_metrics = tf.summary.create_file_writer(tensorboard_logs + '/train/custom_metrics')

    def on_train_begin(self, logs=None):
        # Ensure the custom_metric_string is a valid string
        architecture = getattr(self.model, 'architecture', "architecture")
        
        # Convert the string to a tensor with dtype=tf.string
        custom_metric_tensor = tf.convert_to_tensor(architecture, dtype=tf.string)
        
        # Log to the training log
        with self.custom_writer.as_default():
            tf.summary.text('architecture', custom_metric_tensor, step=0)
        
    def on_epoch_end(self, epoch, logs=None):
        self.learning_rate = logs.get('learning_rate') if logs else None
        self.loss = logs.get('loss') if logs else None

        if self.learning_rate is None and hasattr(self.model.optimizer, 'learning_rate'):
            self.learning_rate = self.model.optimizer.learning_rate

        if self.learning_rate is not None:
            # Log to the training log
            with self.custom_writer_metrics.as_default():
                tf.summary.scalar('learning rate', data=self.learning_rate, step=epoch)
       
        if self.loss is not None:
            with self.custom_writer_metrics.as_default():
                tf.summary.scalar('custom_loss', data=self.loss, step=epoch)

        self.custom_writer_metrics.flush()

           
# Define the custom loss function
# Penalizamos más los errores de las predicciones donde y_true se acerca a su valor minimo, es una exponencial decreciente, se puede trastear con los factores de escala y sesgo, mirar desmos

class CustomLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="custom_loss", scale_factor=2.0, bias_factor=1.0):
        super().__init__(reduction=reduction,name=name)
        self.scale_factor = scale_factor
        self.bias_factor = bias_factor

    def call(self, y_true, y_pred):
        # Calculate the MSE
        mse = tf.square(y_true - y_pred)
        
        # Find the min and max of y_true
        min_y_true = tf.reduce_min(y_true)
        max_y_true = tf.reduce_max(y_true)
        middle_value = (min_y_true + max_y_true) / 2
        range_y_true = max_y_true - min_y_true
        half_range = range_y_true / 2
        
        # Compute normalized distance from middle_value
        normalized_distance = tf.abs(y_true - middle_value) / half_range  # values from 0 to 1
        
        # Apply penalty factor: higher at extremes, lower in the middle
        penalty_factor = (normalized_distance) ** self.scale_factor + self.bias_factor
        
        # Adjust MSE with penalty_factor
        adjusted_mse = mse * penalty_factor
        
        return tf.reduce_mean(adjusted_mse)

class AsymmetricMSELoss(Loss):
    def __init__(self, underestimation_weight=2.0, reduction=tf.keras.losses.Reduction.AUTO, name="asymmetric_mse_loss"):
        super().__init__(reduction=reduction, name=name)
        self.underestimation_weight = tf.constant(underestimation_weight, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # Calculate the error
        error = y_true - y_pred
        
        # Penalize underestimation more by applying a weight
        weight = tf.where(error > 0, self.underestimation_weight, tf.constant(1.0, dtype=tf.float32))
        
        # Calculate the MSE with the asymmetric weighting
        mse = tf.reduce_mean(weight * tf.square(error))
        
        # Return MSE^2
        return mse

# This CustomLoss class combines the three approaches:
# 1. MSE loss as the base.
# 2. A penalty for variance/curvature differences between true and predicted values.
# 3. Emphasis on errors near peaks or valleys in the true data.

    
class NeuralNetworkLayers():
    def dense_layers(x, units, dropout_rate=0):
        x = layers.Dense(units, activation='relu')(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def conv_layers(x, filters, kernel_size, dropout_rate=0, use_batch_norm=False):
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
    
        # 2. Optional Batch Normalization
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        # 3. Optional Dropout Layer
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        
        return x

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
        #x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        res = layers.Add()([x, inputs])

        # Feed Forward Part
        #x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

        return layers.Add()([x, res])
        #return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, res]))
    
class NeuralNetworkModels():
    def ANN_model(mean_extension, 
                  preconcat_mlp_units=[1024, 512, 256], 
                  preconcat_mlp_dropout=[0.3, 0.3, 0.3],
                  postconcat_mlp_units=[1024, 512, 256],
                  postconcat_mlp_dropout=[0.3, 0.3 ,0.3]):

        input_1 = layers.Input(shape=(mean_extension, 2), name='input1')
        input_2 = layers.Input(shape=(1,), name='input2')

        #x = layers.Masking(mask_value=-999)(input_1)
        #x = layers.BatchNormalization()(x)

        x = input_1
        x = layers.concatenate([input_1, layers.RepeatVector(mean_extension)(input_2)], axis=-1)

        for units, dropout_rate in zip(preconcat_mlp_units, preconcat_mlp_dropout):
            x = NeuralNetworkLayers.dense_layers(x, units, dropout_rate)

        z = layers.GlobalMaxPooling1D()(x)
        #x = layers.GlobalAveragePooling1D()(x)

        for units, dropout_rate in zip(postconcat_mlp_units, postconcat_mlp_dropout):
            z = NeuralNetworkLayers.dense_layers(z, units, dropout_rate)

        z = layers.Dense(1, activation='linear', name='output')(z)

        model = Model(inputs=[input_1, input_2], outputs=z, name='ANN_model')

        return model

    def Conv_model(mean_extension, 
                conv_filters=[256, 128], 
                conv_kernels=[3, 3], 
                conv_dropout=[0.2, 0.2],
                use_batch_norm=[True, True],
                mlp_units=[512, 256],
                mlp_dropout=[0.2, 0.2],
                output_dim=1001):


        input_1 = layers.Input(shape=(mean_extension, 2), name='input1')
        input_2 = layers.Input(shape=(1,), name='input2')

        x = input_1
        x = layers.concatenate([input_1, layers.RepeatVector(mean_extension)(input_2)], axis=-1)

        # Apply pre-concatenation convolutions
        for filters, kernel_size, dropout_rate, use_batch_norm in zip(conv_filters, conv_kernels, conv_dropout, use_batch_norm):
            x = NeuralNetworkLayers.conv_layers(x, filters, kernel_size, dropout_rate, use_batch_norm)

        z = layers.GlobalMaxPooling1D()(x)
        #z = layers.concatenate([x, input_2])
        
        for units, dropout_rate in zip(mlp_units, mlp_dropout):
            z = NeuralNetworkLayers.dense_layers(z, units, dropout_rate)

        z = layers.Dense(output_dim, activation='linear', name='output1')(z)  # Output1

        model = Model(inputs=[input_1, input_2], outputs=z, name='Conv_model')

        return model

    def LSTM_model(mean_extension, 
        preconcat_lstm_units=[256, 128, 64],
        mlp_units=[512, 256],
        mlp_dropout=[0.2, 0.2]):

        input_1 = layers.Input(shape=(mean_extension, 2), name='input1')
        input_2 = layers.Input(shape=(1,), name='input2')

        x = input_1
        x = layers.concatenate([input_1, layers.RepeatVector(mean_extension)(input_2)], axis=-1)

        for i, units in enumerate(preconcat_lstm_units):
            # Check if it's the last layer
            return_sequences = True if i < len(preconcat_lstm_units) - 1 else False
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=return_sequences))(x)

        #for units, dropout_rate in zip(mlp_units, mlp_dropout):
        #    x = NeuralNetworkLayers.dense_layers(x, units, dropout_rate)

        x = layers.Dense(1, activation='linear', name='output')(x)

        model = Model(inputs=[input_1, input_2], outputs=x, name='LSTM_model')

        return model

    def Transformer_model(mean_extension, 
                        head_size=[64,64], 
                        num_heads=[4,4], 
                        ff_dim=[1024, 1024],
                        dropout=[0.2, 0.2], 
                        mlp_units=[512,256], 
                        mlp_dropout=[0.3, 0.3],
                        output_dim=1):

        #EL MODELO AL SER UN BUCLE DE TRANSFORMER NECESITA QUE EL INPUT EN CADA BLOQUE SEA LA CONDICION INICIAL, SIN EMBARGO EN CADA BLOQUE ES EL OUTPUT DEL OTRO, HAY QUE FIXEAR
    

        input_1 = layers.Input(shape=(mean_extension, 2), name='input1')
        input_2 = layers.Input(shape=(1,), name='input2')
        
        x = input_1
        x = layers.concatenate([input_1, layers.RepeatVector(mean_extension)(input_2)], axis=-1)

        for head_size, num_heads, ff_dim, dropout in zip(head_size, num_heads, ff_dim, dropout):
            x = NeuralNetworkLayers.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        
        # Global Max and Average Pooling
        #x_max = layers.GlobalMaxPooling1D()(x)
        #x_avg = layers.GlobalAveragePooling1D()(x)
        #x = layers.concatenate([x_max, x_avg]) # Combine both pooling strategies
            
        x = layers.GlobalMaxPooling1D(data_format='channels_first')(x)

        for units, dropout_rate in zip(mlp_units, mlp_dropout):
            x = NeuralNetworkLayers.dense_layers(x, units, dropout_rate)

        x = layers.Dense(output_dim, activation='linear', name='output')(x)

        # Define the model
        model = Model(inputs=[input_1, input_2], outputs=x, name='Transformer_model')

        return model