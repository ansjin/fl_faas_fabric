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

input_size = 784
hidden_size = 500
output_size = 10
input_shape = (28, 28, 1)   

train_images_url = "https://storage.googleapis.com/mnist_fl/mnist_train_img.obj"
train_labels_url = "https://storage.googleapis.com/mnist_fl/mnist_train_labels.obj"

proxies = {
 "http": "http://proxy.in.tum.de:8080/",
 "https": "http://proxy.in.tum.de:8080/",
 "ftp": "ftp://proxy.in.tum.de:8080/"
}

class Client:
    client_id = 0

    def __init__(self, model, data_client):
        train_images = pickle.loads(requests.get(train_images_url, allow_redirects=True, proxies=proxies).content)
        train_labels = pickle.loads(requests.get(train_labels_url, allow_redirects=True, proxies=proxies).content)
        # self.all_samples = [i for i, d in enumerate(train_labels) if d in digits]
        # self.digits = digits
        self.all_samples = self.get_data_from_server(data_client)

        self.X = train_images[self.all_samples]
        self.y = train_labels[self.all_samples]
        self.X = self.X / 255.0

        if model == "cnn":
            self.X  = self.X .reshape(self.X .shape[0], 28, 28, 1)
        elif model == "mnistnn":
            self.X = self.X.reshape(-1, 28*28)
            
        # self.datapoints = 2500

    def create_model(self, hidden_size, input_size, output_size):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
            keras.layers.Dense(output_size)
        ])
        model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.model = model

    def get_data_from_server(self, data_client):
        if(data_client):
            indices = np.fromiter(data_client, np.int32)
            return indices
        else:
            print("Data for the client, does not exist")


    def create_model_cnn(self, input_shape, output_size):
        # model = tf.keras.models.Sequential([
        #         keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        #         keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        #         keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #         keras.layers.Dropout(0.25),
        #         keras.layers.Flatten(),
        #         keras.layers.Dense(128, activation='relu'),
        #         keras.layers.Dropout(0.5),
        #         keras.layers.Dense(output_size)
        # ])
        # model = tf.keras.models.Sequential([
        #         keras.layers.Conv2D(10, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        #         keras.layers.Conv2D(20, kernel_size=(5, 5), activation='relu'),
        #         keras.layers.Flatten(),
        #         keras.layers.Dense(320, activation='relu'),
        #         keras.layers.Dense(output_size)
        # ])
        model = tf.keras.models.Sequential([
                keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(output_size)
        ])

        model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.model = model

    def get_model_weights_cardinality(self):
        return self.model.get_weights(), self.cardinality
     
    def set_model_weights(self, weights):
        # self.model.set_weights(list(weights.values()))
        self.model.set_weights(weights)
    
    def create_datasetObject(self, batch_size=10):
        # dataset_train = tf.data.Dataset.from_tensor_slices((self.X[self.all_samples], self.y[self.all_samples]))
        dataset_train = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        dataset_train = dataset_train.shuffle(len(self.y))
        dataset_train = dataset_train.batch(batch_size)
        # dataset_val = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        # dataset_val = dataset_val.shuffle(len(self.y))
        self.dataset_train = dataset_train
        self.cardinality = tf.data.experimental.cardinality(self.dataset_train).numpy()*batch_size
        print("Cardinality Client {0} is {1}".format(self.client_id, tf.data.experimental.cardinality(self.dataset_train).numpy()))
        return self.dataset_train

    
    def request_update(self, epochs=5, batch_size=10):
        # X = self.X[self.all_samples]
        # y = self.y[self.all_samples]
        dataset_train  = self.create_datasetObject(batch_size)
        # X_val = 
        # y_val = self.y_val
        self.model.fit(dataset_train,
          epochs=epochs,
          batch_size=batch_size
        )  
    



def get_config():
    config = {}
    config['input_size'] = input_size
    config['hidden_size'] = hidden_size
    config['output_size'] = output_size
    config['input_shape'] = input_shape
    # config['client_keys'] = client_keys

    return config


def get_weights_from_server(server_weights):
    weights = []
    if(server_weights):
        weights = pickle.loads(server_weights)
    else:
        print("No weights for Server exist, initialize it first")
    return weights


def write_updated_weights_client(weights,cardinality):
    weights_serialized = Binary(pickle.dumps(weights, protocol=2), subtype=128)
    new_values = {'weights':weights_serialized, 'cardinality': int(cardinality)}
    #print("Data updated with id {0} for Client {1}".format(new_values, client_id))
    return new_values

def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf


def main(request):
    request_json = bson.BSON(request.data).decode()

    try:
        server_weights = request_json['server']
        data_client = request_json['client']

    except:
        return {'Error' : 'Input parameters should include a string to sentiment analyse.'}

    config = get_config()

    # train_images, train_labels = load_mnist_data(config['train_images'], config['train_labels'])
    #train_images = train_images.astype(np.float32)
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    client = Client("cnn", data_client)
    # client.create_model(config['hidden_size'], config['input_size'], config['output_size'])
    client.create_model_cnn(config['input_shape'],  config['output_size'])
   
    server_weights = get_weights_from_server(server_weights)
    client.set_model_weights(server_weights)

    client.request_update()
    updated_weights, cardinality = client.get_model_weights_cardinality()

    new_weights = write_updated_weights_client(updated_weights, cardinality)

    new_weights = bson.BSON.encode(new_weights)
    
    return dumps(new_weights)


if __name__ == "__main__":
    st = get_stdin()
    print(main(st))
