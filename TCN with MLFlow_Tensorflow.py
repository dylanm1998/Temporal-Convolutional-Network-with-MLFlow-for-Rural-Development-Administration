import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from bson.binary import Binary
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Layer, Conv1D, Input, Dense, SpatialDropout1D, BatchNormalization, Lambda, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau

class MongoDatabase:
    def __init__(self):
        CONNECTION_STRING = "mongodb://netdb:netdb3230!@10.255.93.173:27017/"
        self.client = MongoClient(CONNECTION_STRING)

    def _fetch_data(self, collection_name, limit=None):
        """Private method to fetch data from a specified collection in MongoDB."""
        try:
            collection = self.client["TestAPI"][collection_name]
            cursor = collection.find({}).limit(limit) if limit else collection.find({})
            return pd.DataFrame(list(cursor))
        except Exception as e:
            print(f"Error while fetching data from {collection_name}: {e}")
            return None

    def get_environment(self, limit=None):
        """Public method to fetch environment data from the 'GH2' collection."""
        return self._fetch_data("GH2", limit)

    def get_growth(self, limit=None):
        """Public method to fetch growth data from the 'hydroponics_length1' collection."""
        return self._fetch_data("hydroponics_length1", limit)

    def save_model(self, model, model_name, model_type):
        """Method to save a model to MongoDB. It saves the model's HDF5 file."""
        model_file = f"{model_name}.h5"
        model.save(model_file)

        # Read and store the HDF5 file data
        with open(model_file, 'rb') as file:
            model_data = file.read()

        db = self.client["Things_to_refer"]
        collection = db["Previous_model_features"]

        # Create a document with model information
        model_document = {
            "name": model_name,
            "type": model_type,
            "model_data": Binary(model_data)
        }

        # Check if a model with the same name exists and update it, else insert a new document
        existing_document = collection.find_one({"name": model_name})
        if existing_document:
            collection.update_one({"_id": existing_document["_id"]}, {"$set": model_document})
            print(f"Existing model '{model_name}' updated in MongoDB.")
        else:
            collection.insert_one(model_document)
            print(f"New model '{model_name}' inserted into MongoDB.")

    def load_model(self, model_name):
        """Method to load a model from MongoDB."""
        try:
            db = self.client["Things_to_refer"]
            collection = db["Previous_model_features"]
            model_document = collection.find_one({"name": model_name})
            
            if model_document:
                model_data = model_document["model_data"]
                with open(f"{model_name}.h5", 'wb') as file:
                    file.write(model_data)
                model = tf.keras.models.load_model(f"{model_name}.h5")
                print(f"Model '{model_name}' loaded from MongoDB.")
                return model
            else:
                print(f"No model found with the name '{model_name}'.")
                return None
        except Exception as e:
            print(f"Error while loading model '{model_name}': {e}")
            return None

def create_dataset(X, y, look_back=1):
    """
    Create dataset for time-series forecasting.
    
    Parameters:
    - X: Input time-series data (features), a 2D NumPy array where rows represent time steps and columns represent features.
    - y: Output time-series data (target), a 1D or 2D NumPy array where rows represent time steps.
    - look_back (default=1): Number of previous time steps to use as input variables to predict the next time step.
    
    Returns:
    - dataX: 3D NumPy array of the input sequences, shape (num_samples, look_back, num_features).
    - dataY: 1D or 2D NumPy array of the output sequences, shape (num_samples, num_output_features).
    """
    
    dataX, dataY = [], []  # Initialize empty lists to hold our transformed sequences.
    
    # For each possible sequence in the input data...
    for i in range(len(X) - look_back):
        # Extract a sequence of 'look_back' features from the input data.
        sequence = X[i:(i + look_back), :]
        dataX.append(sequence)
        
        # Extract the output for this sequence from the 'y' data.
        output = y[i + look_back]
        dataY.append(output)

    # Convert the lists into NumPy arrays for compatibility with most ML frameworks.
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # Log the shape of the created datasets for debugging
    print(f"Input sequence shape: {dataX.shape}")
    print(f"Output sequence shape: {dataY.shape}")

    return dataX, dataY

# TCN Model
def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)

def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations
    
class ResidualBlock(Layer):
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, activation='relu',
                 dropout_rate=0, kernel_initializer='he_normal', use_batch_norm=False,
                 use_layer_norm=False, use_weight_norm=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer

        self.conv_layers = []
        for k in range(2):
            self.conv_layers.append(Conv1D(filters=self.nb_filters,
                                           kernel_size=self.kernel_size,
                                           dilation_rate=self.dilation_rate,
                                           padding=self.padding,
                                           kernel_initializer=self.kernel_initializer))
            if self.use_batch_norm:
                self.conv_layers.append(BatchNormalization())
            if self.use_layer_norm:
                self.conv_layers.append(LayerNormalization())
            self.conv_layers.append(Activation(self.activation))
            self.conv_layers.append(SpatialDropout1D(rate=self.dropout_rate))
        
        if self.nb_filters != self.kernel_size:
            self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                           kernel_size=1,
                                           padding='same',
                                           kernel_initializer=self.kernel_initializer)
        else:
            self.shape_match_conv = Lambda(lambda x: x)
        
        self.final_activation = Activation(self.activation)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x, training=training)
        
        res_x = self.shape_match_conv(inputs)
        x += res_x
        return self.final_activation(x)
    
class TCN(Layer):
    def __init__(self, nb_filters=64, kernel_size=3, nb_stacks=1, dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal', use_skip_connections=True, dropout_rate=0.0,
                 return_sequences=False, activation='relu', kernel_initializer='he_normal',
                 use_batch_norm=False, use_layer_norm=False, use_weight_norm=False,
                 go_backwards=False, return_state=False, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.go_backwards = go_backwards
        self.return_state = return_state

        self.residual_blocks = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=self.nb_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation_name,
                                                          dropout_rate=self.dropout_rate,
                                                          kernel_initializer=self.kernel_initializer,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm))

    def call(self, inputs, training=None):
        x = inputs
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        if self.return_sequences:
            return x
        else:
            return x[:, -1, :]
        
def compiled_tcn(num_feat, num_classes, nb_filters, kernel_size, dilations, nb_stacks, max_len,
                 output_len=1, padding='causal', use_skip_connections=False, return_sequences=True,
                 regression=False, dropout_rate=0.05, name='tcn', kernel_initializer='he_normal',
                 activation='relu', opt='adam', lr=0.002, use_batch_norm=False,
                 use_layer_norm=False, use_weight_norm=False):
    dilations = adjust_dilations(dilations)
    input_layer = Input(shape=(max_len, num_feat))
    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding, use_skip_connections,
            dropout_rate, return_sequences, activation, kernel_initializer, use_batch_norm,
            use_layer_norm, use_weight_norm, name=name)(input_layer)

    def get_opt():
        if opt == 'adam':
            return Adam(learning_rate=lr, clipnorm=1.0)
        elif opt == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
        else:
            raise ValueError('Only Adam and RMSProp are available here')

    if not regression:
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=get_opt(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=get_opt(), loss='mean_squared_error')

    return model

def tcn_full_summary(model: Model, expand_residual_blocks=True):
    if tf.__version__ <= '2.5.0':
        layers = model.layers.copy()
        model._layers.clear()

        for layer in layers:
            if isinstance(layer, TCN):
                for sub_layer in layer.layers:
                    if not isinstance(sub_layer, ResidualBlock):
                        model._layers.append(sub_layer)
                    else:
                        if expand_residual_blocks:
                            for sub_sub_layer in sub_layer.layers:
                                model._layers.append(sub_sub_layer)
                        else:
                            model._layers.append(sub_layer)
            else:
                model._layers.append(layer)

        model.summary()
        model._layers.clear()
        model._layers.extend(layers)
    else:
        print('WARNING: tcn_full_summary: Compatible with tensorflow 2.5.0 or below.')
        print('Use tensorboard instead.')

def Save_model(model, model_name, root_folder="saved_models"):
    """
    Save a given model's architecture as a JSON file and weights as an H5 file.
    
    Parameters:
    - model: Trained model to save.
    - model_name: Name of the model (e.g., "LSTM", "RNN").
    - root_folder (default='saved_models'): Name of the root folder where model subfolders will be created.
    
    Returns:
    - None
    """
    # Define the model-specific directory path
    model_dir = os.path.join(root_folder, model_name)

    # Ensure the save directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory {model_dir} for saving the model.")

    # Save the model architecture as a JSON file
    model_json_path = os.path.join(model_dir, f"{model_name}.json")
    with open(model_json_path, "w") as json_file:
        json_file.write(model.to_json())
    print(f"Model architecture saved to {model_json_path}")

    # Save the model weights as an H5 file
    model_weights_path = os.path.join(model_dir, f"{model_name}.weights.h5")
    model.save_weights(model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    print(f"Saved {model_name} model to {model_dir}.")

def train_TCN_model(X_train, Y_train, X_val, Y_val, look_back=10, num_feat=2, nb_filters=64, kernel_size=4, dilations=[1, 2, 4, 8, 16, 32], nb_stacks=1, lr=0.005, dropout_rate=0.5):
    model = compiled_tcn(num_feat=num_feat, num_classes=1, nb_filters=nb_filters, kernel_size=kernel_size,
                         dilations=dilations, nb_stacks=nb_stacks, max_len=look_back, regression=True,
                         dropout_rate=dropout_rate, lr=lr)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32, callbacks=[early_stop, reduce_lr])
    
    return model, history


# Create an instance of the MongoDatabase class
db = MongoDatabase()

# Fetch growth data using the 'get_growth' method from the 'db' object
growth_data_1 = db.get_growth()
# print("Original Growth Data:")
# print(growth_data_1)

growth_data_2 = growth_data_1.drop(columns=['_id', 'date', 'sample_num', 'plant_height              (㎝)', 'plant_diameter           (㎜)', 'leaflet          (cm)', 'leaf_width         (cm)', 'last_flower_point         (th)', 'growing_point_to_flower_point        (㎝)', 'note'], errors='ignore')
# print("Processed Growth Data:")
# print(growth_data_2)

# Fetch environment data using the 'get_environment' method from the 'db' object.
environment_data_1 = db.get_environment(limit = 31200)
# print("Original Environment Data:")
# print(environment_data_1)

# Modify the 'environment_data_1' DataFrame to drop specified columns.
# environment_data_2 = environment_data_1.drop(columns=['_id', 'id', 'inFacilityId', 'sensorNo', 'sensingAt'], errors='ignore')
environment_data_2 = environment_data_1.drop(columns=['_id', 'id', 'inFacilityId', 'sensorNo', 'sensingAt', 'co2'], errors='ignore')
# print("Processed Environment Data:")
# print(environment_data_2)

environment_averaged = environment_data_2.groupby(environment_data_2.index // 100).mean(numeric_only=True).reset_index(drop=True)
# print("Averaged Environment Data:")
# print(environment_averaged)

# Merge the 'environment_averaged' DataFrame and 'growth_data_2' DataFrame based on their indices.
training_data = pd.merge(environment_averaged, growth_data_2, left_index=True, right_index=True)
# print("Merged Training Data:")
# print(training_data)

# Initialize the MinMaxScaler.
scaler = MinMaxScaler()
# 'data_normalized' will be a NumPy array where each feature (column) of the input data is normalized to the range [0, 1].
data_normalized = scaler.fit_transform(training_data)
# print("Normalized Training Data:")
# print(data_normalized)

# Assuming the last column of 'data_normalized' is the target variable that want to predict.
# 'data_normalized' is a 2D array with rows as individual data records and columns as features.

# Extract input features (every column except the last one).
X_data = data_normalized[:, :-1]

# Extract target variable (just the last column).
y_data = data_normalized[:, -1]

# Define the look-back period, which determines the number of past observations 
# each input sequence will contain when transforming the data.
look_back = 10

# Transform the data into sequences of input (X) and output (Y) using the 'create_dataset' function.
X, Y = create_dataset(X_data, y_data, look_back)

# Define the size of the training set as 80% of the total data.
train_size = int(len(X) * 0.8)

# Split the data based on order (important for time series data).
# The first 80% is used for training.
X_train, X_temp = X[:train_size], X[train_size:]
Y_train, Y_temp = Y[:train_size], Y[train_size:]

# The remaining 20% is further divided into validation and test sets, each taking 10%.
# Split the remaining data into half for validation and testing.
val_size = len(X_temp) // 2

# Extract validation and test sets from the remaining data.
X_val, X_test = X_temp[:val_size], X_temp[val_size:]
Y_val, Y_test = Y_temp[:val_size], Y_temp[val_size:]

# Print shapes to verify the splits
# print(f"Training data shape: X_train={X_train.shape}, Y_train={Y_train.shape}")
# print(f"Validation data shape: X_val={X_val.shape}, Y_val={Y_val.shape}")
# print(f"Test data shape: X_test={X_test.shape}, Y_test={Y_test.shape}")

look_back = 10
num_feat = 2

mlflow.set_experiment("TF_TCN_Model")

with mlflow.start_run():
    mlflow.tensorflow.autolog()  # Automatically record TensorFlow parameters, indicators and models

    # Train TCN model
    TCN_model, TCN_history = train_TCN_model(X_train, Y_train, X_val, Y_val, look_back=look_back, num_feat=num_feat)

    # Predict test set
    predicted_values = TCN_model.predict(X_test)
    predicted_values = np.squeeze(predicted_values)

    # Check if the shapes match
    print(f"Shape of predicted_values: {predicted_values.shape}")
    print(f"Shape of Y_test: {Y_test.shape}")

    # Average predicted_values
    predicted_values = np.mean(predicted_values, axis=1)

    # Visualize predictions vs true values(In testing)
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, predicted_values, color='blue', alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig("Actual_vs_Predicted_values.png")
    plt.close()

     # Log scatter plot to mlflow
    mlflow.log_artifact("Actual_vs_Predicted_values.png")

    # Plot a comparison between predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, label="Actual values", color='blue', alpha=0.5)
    plt.plot(predicted_values, label="Predicted values of ", color='red', alpha=0.5)
    plt.title("Predicted_values vs Actual values")
    plt.savefig("Comparison_plot.png")
    plt.close() 

    # Log comparison plot to mlflow
    mlflow.log_artifact("Comparison_plot.png")

    mse_tcn = mean_squared_error(Y_test, predicted_values)
    rmse_tcn = np.sqrt(mse_tcn)
    mae_tcn = mean_absolute_error(Y_test, predicted_values)
    
    # Logging metrics into MLflow
    mlflow.log_metric("mse_tcn", mse_tcn)
    mlflow.log_metric("rmse_tcn", rmse_tcn)
    mlflow.log_metric("mae_tcn", mae_tcn)
    
    # Print metrics
    # print(f"MSE: {mse_tcn}")
    # print(f"RMSE: {rmse_tcn}")
    # print(f"MAE: {mae_tcn}")

mlflow.end_run()