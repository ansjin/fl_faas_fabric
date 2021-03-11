import sys
import json
import numpy as np
from bson.binary import Binary
import pickle
import requests
import tensorflow as tf 
from tensorflow import keras
import os
import bson
from bson.json_util import dumps
np.random.seed(2)

class Client:
    client_id = 0

    def __init__(self, lr, optim, local_epochs, model, server_weights, data_client,
                 train_images_url,
                 train_labels_url,
                 test_images_url,
                 test_labels_url,
                 input_size,
                 hidden_size,
                 output_size,
                 input_shape_x,
                 input_shape_y,
                 batch_size,
                 ):

        self.proxies = {
            "http": "http://proxy.in.tum.de:8080/",
            "https": "http://proxy.in.tum.de:8080/",
            "ftp": "ftp://proxy.in.tum.de:8080/",
            "no_proxy": "172.24.65.16"
        }
        self.data_client = data_client
        self.server_weights = server_weights

        self.train_images_url = train_images_url
        self.train_labels_url = train_labels_url
        self.test_images_url = test_images_url
        self.test_labels_url = test_labels_url

        self.config = {}
        self.config['input_size'] = input_size
        self.config['hidden_size'] = hidden_size
        self.config['output_size'] = output_size
        self.config['input_shape'] = (input_shape_x, input_shape_y, 1)
        self.config['batch_size'] = batch_size
        self.config['model'] = model
        self.config['lr'] = lr
        self.config['optim'] = optim
        self.config['local_epochs'] = local_epochs

        train_images = pickle.loads(requests.get(self.train_images_url, allow_redirects=True).content)
        train_labels = pickle.loads(requests.get(self.train_labels_url, allow_redirects=True).content)

        self.all_samples = self.get_data_from_server()

        self.X = train_images[self.all_samples]
        self.y = train_labels[self.all_samples]
        self.y = self.y.astype('int32')


        self.X = self.X / 255.0

        if self.config['model'] == "cnn":
            self.X  = self.X .reshape(self.X .shape[0], input_shape_x, input_shape_y, 1)
        elif self.config['model'] == "mnistnn":
            self.X = self.X.reshape(-1, input_shape_x * input_shape_y)
            
        # self.datapoints = 2500

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(self.config['hidden_size'], activation='relu', input_shape=(self.config['input_size'],)),
            keras.layers.Dense(self.config['output_size'])
        ])

        optim = self.config['optim']
        lr = float(self.config['lr'])
        if optim == "adam":
            opt = tf.keras.optimizers.Adam(lr=lr)
        elif optim == "ndam":
            opt = tf.keras.optimizers.Nadam(lr=lr)
        elif optim == "sgd":
            opt = tf.keras.optimizers.SGD(lr=lr)

        model.compile(optimizer=opt,
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.model = model

    def get_data_from_server(self):
        if(self.data_client):
            indices = np.fromiter(self.data_client, np.int32)
            return indices
        else:
            print("Data for the client, does not exist")


    def create_model_cnn(self):

        model = tf.keras.models.Sequential([
                keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=self.config['input_shape']),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(self.config['output_size'])
        ])

        optim = self.config['optim']
        lr = float(self.config['lr'])
        if optim == "adam":
            opt = tf.keras.optimizers.Adam(lr=lr)
        elif optim == "ndam":
            opt = tf.keras.optimizers.Nadam(lr=lr)
        elif optim == "sgd":
            opt = tf.keras.optimizers.SGD(lr=lr)

        model.compile(optimizer=opt,
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.model = model

    def get_model_weights_cardinality(self):
        return self.model.get_weights(), self.cardinality
     
    def set_model_weights(self, weights):
        # self.model.set_weights(list(weights.values()))
        self.model.set_weights(weights)
    
    def create_datasetObject(self):
        # dataset_train = tf.data.Dataset.from_tensor_slices((self.X[self.all_samples], self.y[self.all_samples]))
        dataset_train = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        dataset_train = dataset_train.shuffle(len(self.y))
        dataset_train = dataset_train.batch(self.config['batch_size'])
        # dataset_val = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        # dataset_val = dataset_val.shuffle(len(self.y))
        self.dataset_train = dataset_train
        self.cardinality = tf.data.experimental.cardinality(self.dataset_train).numpy()*self.config['batch_size']
        print("Cardinality Client {0} is {1}".format(self.client_id, tf.data.experimental.cardinality(self.dataset_train).numpy()))
        return self.dataset_train

    
    def request_update(self):
        # X = self.X[self.all_samples]
        # y = self.y[self.all_samples]
        dataset_train  = self.create_datasetObject()
        # X_val = 
        # y_val = self.y_val
        epochs = int(self.config['local_epochs'])

        self.model.fit(dataset_train,
          epochs=epochs,
          batch_size=self.config['batch_size']
        )  

    def get_weights_from_server(self):
        weights = []
        if(self.server_weights):
            weights = pickle.loads(self.server_weights)
        else:
            print("No weights for Server exist, initialize it first")
        return weights

    def write_updated_weights_client(self, weights, cardinality):
        weights_serialized = Binary(pickle.dumps(weights, protocol=2), subtype=128)
        new_values = {'weights':weights_serialized, 'cardinality': int(cardinality)}
        #print("Data updated with id {0} for Client {1}".format(new_values, client_id))
        return new_values


def main(request):
    request_json = bson.BSON(request.data).decode()

    try:
        client_obj = Client(request_json["lr"],
                            request_json["optim"],
                            request_json["local_epochs"],
                            request_json["model"],
                            request_json["server"],
                            request_json["client"],
                            request_json["train_images_url"],
                            request_json["train_labels_url"],
                            request_json["test_images_url"],
                            request_json["test_labels_url"],
                            request_json["data_sampling"]["input_size"],
                            request_json["data_sampling"]["hidden_size"],
                            request_json["data_sampling"]["output_size"],
                            request_json["data_sampling"]["input_shape_x"],
                            request_json["data_sampling"]["input_shape_y"],
                            request_json["data_sampling"]["batch_size"])

    except:
        return {'Error' : 'Input parameters should include a string to sentiment analyse.'}

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # client.create_model(config['hidden_size'], config['input_size'], config['output_size'])

    if client_obj.config['model'] == "cnn":
        client_obj.create_model_cnn()
    else:
        client_obj.create_model()
   
    server_weights_updated = client_obj.get_weights_from_server()
    client_obj.set_model_weights(server_weights_updated)

    client_obj.request_update()
    updated_weights, cardinality = client_obj.get_model_weights_cardinality()

    new_weights = client_obj.write_updated_weights_client(updated_weights, cardinality)

    new_weights = bson.BSON.encode(new_weights)
    
    return dumps(new_weights)
