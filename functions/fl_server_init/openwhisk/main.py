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


class FLServerInit:

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
        self.config['input_shape'] = (input_shape_x, input_shape_y, 1)
        self.config['batch_size'] = batch_size
        self.config['model'] = model

        client = MongoClient(self.mongo_url)
        db = client[self.mongo_db]
        self.collection = db[self.collection_name]

    def mnist_noniid(self):
        # X_train = pickle.load(open(filename_X_train, 'rb'))
        # y_train = pickle.load(open(filename_y_train, 'rb'))
        y_train = pickle.loads(requests.get(self.train_labels_url, allow_redirects=True, proxies=self.proxies).content)
        num_shards, num_imgs = 200, 300
        idx_shard = [i for i in range(num_shards)]
        # print(idx_shard)
        dict_users = {str(i): np.array([], dtype=np.int64) for i in range(self.config['num_clients'])}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = y_train


        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]


        # divide and assign 2 shards/client
        for i in range(self.config['num_clients']):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            # print(rand_set)
            idx_shard = list(set(idx_shard) - rand_set)
            # print(idx_shard)
            for rand in rand_set:
                # print(dict_users[i])
                dict_users[str(i)] = np.concatenate(
                    (dict_users[str(i)], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        return dict_users

    def write_weights_to_server(self, weights, key='Server'):
        document = self.collection.find_one({'key': key})
        weights_serialized = Binary(pickle.dumps(weights, protocol=2), subtype=128)
        if(document):
            filter = {'key': key}
            new_values = {'$set': {'weights':weights_serialized}}
            self.collection.update_one(filter, new_values)
            # print("Data updated with id {0}".format(result))
        else:
            values = {'key': key,'weights':weights_serialized}
            self.collection.insert_one(values)
            # print("No weights for Server exist, Added with id {0}".format(result.inserted_id))

    def get_weights_from_server(self, key='Server'):
        document = self.collection.find_one({'key': key})
        weights = []
        if(document):
            weights = pickle.loads(document['weights'])
        return weights

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

    def get_server_test_data(self):
        # print(X_test.shape)
        # print(y_test.shape)
        X_test = pickle.loads(requests.get(self.test_images_url, allow_redirects=True, proxies=self.proxies).content)
        y_test = pickle.loads(requests.get(self.test_labels_url, allow_redirects=True, proxies=self.proxies).content)

        return X_test, y_test

    def evaluate_model(self, model, updated_weights):
        # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        X_test, y_test = self.get_server_test_data()
        # test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
        X_test = X_test.reshape(-1, 28*28)
        X_test = X_test / 255.0
        model.set_weights(updated_weights)
        loss,acc = model.evaluate(X_test,  y_test, batch_size=self.config['batch_size'] ,verbose=2)
        # print("Restored model, accuracy: {:.3f}%".format(100*acc))
        return loss, acc

    def write_data_dict_to_server(self):

        data_clients  = self.mnist_noniid()
        keys = ["data_client_" + str(i) for i in range(self.config['num_clients'])]
        for i in range(len(keys)):
            document = self.collection.find_one({'key': keys[i]})
            if(document):
                filter = {'key': keys[i]}
                new_values = {'$set': {'data': data_clients[str(i)].tolist()}}
                result = self.collection.update_one(filter, new_values)
                print("Data updated with id {0} for Client {1}".format(result, keys[i]))
            else:
                values = {'key': keys[i], 'data': data_clients[str(i)].tolist()}
                result = self.collection.insert_one(values)
                print("Data inserted with id {0}".format(result.inserted_id))

    def initialize_clients(self):
        keys = ["client_" + str(i) for i in range(self.config['num_clients'])]
        for i in range(len(keys)):
            document = self.collection.find_one({'key': keys[i]})
            if(document):
                filter = {'key': keys[i]}
                new_values = {'$set': {'weights': int(0), 'cardinality': int(0), 'optimizer_weights': int(0)}}
                result = self.collection.update_one(filter, new_values)
            else:
                values = {'key': keys[i], 'weights':int(0), 'cardinality': int(0)}
                result = self.collection.insert_one(values)

def main(params):
    try:
        fl_server_init_obj = FLServerInit("mongodb://" + params["mongo"]["url"] + "/",
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
                                          params["data_sampling"]["model"],)

    except:
        return {'Error': 'Input parameters should include a string to sentiment analyse.'}
        
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    fl_server_init_obj.write_data_dict_to_server()

    if fl_server_init_obj.config['model'] == "cnn":
        server_model = fl_server_init_obj.create_model_cnn()
    else:
        server_model = fl_server_init_obj.create_model()

    fl_server_init_obj.write_weights_to_server(fl_server_init_obj.get_weights(server_model))

    ret_val = {}
    ret_val['weights'] = "done"
    return ret_val
