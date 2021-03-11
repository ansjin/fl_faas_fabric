import sys
import json
from pymongo import MongoClient
import numpy as np
from bson.binary import Binary
import pickle
import struct
import tensorflow as tf
from tensorflow import keras
import requests
import os


class FLServerAgg:

    def __init__(self, mongo_url, mongo_db, collection_name,
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
                 num_clients,
                 model
                 ):
        self.mongo_url = mongo_url
        self.proxies = {
            "http": "http://proxy.in.tum.de:8080/",
            "https": "http://proxy.in.tum.de:8080/",
            "ftp": "ftp://proxy.in.tum.de:8080/",
            "no_proxy": "172.24.65.16"
        }
        self.mongo_db = mongo_db
        self.collection_name = collection_name

        self.train_images_url = train_images_url
        self.train_labels_url = train_labels_url
        self.test_images_url = test_images_url
        self.test_labels_url = test_labels_url

        self.config = {}
        self.config['input_size'] = input_size
        self.config['hidden_size'] = hidden_size
        self.config['output_size'] = output_size
        self.config['num_clients'] = num_clients
        self.config['input_shape_x'] = input_shape_x
        self.config['input_shape_y'] = input_shape_y

        self.config['input_shape'] = (input_shape_x, input_shape_y, 1)
        self.config['batch_size'] = batch_size
        self.config['model'] = model

        client = MongoClient(self.mongo_url)
        db = client[self.mongo_db]
        self.collection = db[self.collection_name]


    def write_weights_to_server(self, weights, key='Server'):
        document = self.collection.find_one({'key': key})
        weights_serialized = Binary(pickle.dumps(weights, protocol=2), subtype=128)
        if (document):
            filter = {'key': key}
            new_values = {'$set': {'weights': weights_serialized}}
            self.collection.update_one(filter, new_values)
            # print("Data updated with id {0}".format(result))
        else:
            values = {'key': key, 'weights': weights_serialized}
            self.collection.insert_one(values)
            # print("No weights for Server exist, Added with id {0}".format(result.inserted_id))

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(self.config['hidden_size'], activation='relu',
                               input_shape=(self.config['input_size'],)),
            keras.layers.Dense(self.config['output_size'])
        ])

        model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        return model

    def create_model_cnn(self):
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
            keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=self.config['input_shape']),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=self.config['input_shape']),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(self.config['output_size'])
        ])

        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    def get_weights(self, model):
        return model.get_weights()

    def get_weights_from_clients(self, keys):
        weights=[]
        for i in range(0, self.config['num_clients']):
            document = self.collection.find_one({'key': keys[i]})
            if(document):
                individual_weights = pickle.loads(document['weights'])
                weights.append(individual_weights)
            else:
                print("Client {0} does not exist".format(keys[i]))
        return weights

    def get_cardinality_from_clients(self, keys):
        cardinalities=[]
        for i in range(0, self.config['num_clients']):
            document = self.collection.find_one({'key': keys[i]})
            if(document):
                # individual_weights = pickle.loads(document['weights'])
                individual_cardinality = document['cardinality']
                cardinalities.append(individual_cardinality)
            else:
                print("Client {0} does not exist".format(keys[i]))
        return cardinalities

    def get_server_test_data(self):
        # print(X_test.shape)
        # print(y_test.shape)
        X_test = pickle.loads(requests.get(self.test_images_url, allow_redirects=True, proxies=self.proxies).content)
        y_test = pickle.loads(requests.get(self.test_labels_url, allow_redirects=True, proxies=self.proxies).content)

        return X_test, y_test

    def scale_model_weights(self, weight, scalar):
            # weight = self.get_model_weights()
            weight_final = []
            steps = len(weight)
            for i in range(steps):
                weight_final.append(scalar * weight[i])
            # self.set_model_weights(weight_final)
            return weight_final

    def fl_average(self, client_weight_list, client_cardinality_list):
        global_training_samples = sum(client_cardinality_list)
        print("Global training samles are {0}".format(global_training_samples))
        scaled_weight_list = []
        avg_grad = list()
        for i in range(0, self.config['num_clients']):
            scaled_weights = self.scale_model_weights(client_weight_list[i],
                                                      float(client_cardinality_list[i]/global_training_samples))
            scaled_weight_list.append(scaled_weights)
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad

    def evaluate_model(self, model, updated_weights):
        # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        X_test, y_test = self.get_server_test_data()
        # test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], self.config['input_shape_x'], self.config['input_shape_y'], 1)

        if self.config["model"] == "mnistnn":
            X_test = X_test.reshape(-1, 28*28)

        X_test = X_test / 255.0
        model.set_weights(updated_weights)
        loss,acc = model.evaluate(X_test,  y_test, batch_size=self.config['batch_size'], verbose=2)
        # print("Restored model, accuracy: {:.3f}%".format(100*acc))
        return loss, acc


def main(params):
    try:
        fl_server_agg_obj = FLServerAgg("mongodb://" + params["mongo"]["url"] + "/",
                                        params["mongo"]["db"],
                                        params["mongo"]["collection"],
                                        params["train_images_url"],
                                        params["train_labels_url"],
                                        params["test_images_url"],
                                        params["test_labels_url"],
                                        params["data_sampling"]["input_size"],
                                        params["data_sampling"]["hidden_size"],
                                        params["data_sampling"]["output_size"],
                                        params["data_sampling"]["input_shape_x"],
                                        params["data_sampling"]["input_shape_y"],
                                        params["data_sampling"]["batch_size"],
                                        params["num_clients"],
                                        params["data_sampling"]["model"],
                                        )

    except:
        return {'Error': 'Input parameters should include a string to sentiment analyse.'}

    # client_keys =["client_" +  str(i) for i in range(0, fl_server_agg_obj.config['num_clients'])]
    client_keys = params["client_round_ids"]

    fl_server_agg_obj.config['client_keys'] = client_keys

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # server_model = create_model(config['hidden_size'], config['input_size'], config['output_size'])
    if fl_server_agg_obj.config['model'] == "cnn":
        server_model = fl_server_agg_obj.create_model_cnn()
    else:
        server_model = fl_server_agg_obj.create_model()


    client_weights_list = fl_server_agg_obj.get_weights_from_clients(fl_server_agg_obj.config['client_keys'])
    cardinalities_client = fl_server_agg_obj.get_cardinality_from_clients(fl_server_agg_obj.config['client_keys'])
    
    updated_weights = fl_server_agg_obj.fl_average( client_weights_list, cardinalities_client)
    loss, acc = fl_server_agg_obj.evaluate_model(server_model, updated_weights)
    fl_server_agg_obj.write_weights_to_server(updated_weights)

    ret_val = {}
    ret_val['loss'] = loss
    ret_val['accuracy'] = acc
    # ret_val['weights'] = "done"
    return ret_val

