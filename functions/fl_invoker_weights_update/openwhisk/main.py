from pymongo import MongoClient
import bson
import requests
from bson.json_util import loads
from bson.json_util import dumps
import json


class FLServerUpdateInvoke:

    def __init__(self, mongo_url, mongo_db, collection_name):
        self.mongo_url = mongo_url
        self.proxies = {
            "http": "http://proxy.in.tum.de:8080/",
            "https": "http://proxy.in.tum.de:8080/",
            "ftp": "ftp://proxy.in.tum.de:8080/",
            "no_proxy": "172.24.65.16"
        }
        self.mongo_db = mongo_db
        self.collection_name = collection_name

        client = MongoClient(self.mongo_url)
        db = client[self.mongo_db]
        self.collection = db[self.collection_name]

    def get_data_from_server(self, client_id_num):
        data_client_id = "data_client_" + str(client_id_num)
        document = self.collection.find_one({'key': data_client_id})
        if(document):
            data = document['data']
            return data
        else:
            print("Data for the client, does not exist")

    def get_weights_from_server(self, key='Server'):
        document = self.collection.find_one({'key': key})
        weights = []
        if(document):
            weights = document['weights']
        return weights

    def write_updated_weights_client(self, weights_serialized, cardinality, key):
        document = self.collection.find_one({'key': key})
        #weights_serialized = Binary(pickle.dumps(weights, protocol=2), subtype=128)
        if document:
            filter = {'key': key}
            new_values = {'$set': {'weights':weights_serialized, 'cardinality': int(cardinality)}}
            result = self.collection.update_one(filter, new_values)
            print("Data updated with id {0} for Client {1}".format(result, key))
        else:
            values = {'key': key, 'weights':weights_serialized, 'cardinality': int(cardinality)}
            result = self.collection.insert_one(values)
            print("Data inserted with id {0}".format(result.inserted_id))



def main(params):
    try:
        client_id = params["client_id"]
        url = params["url"]
        client_type = params["client_type"]
        fl_server_update_invoke_obj = FLServerUpdateInvoke("mongodb://" + params["mongo"]["url"] + "/",
                                                           params["mongo"]["db"],
                                                           params["mongo"]["collection"])

    except:
        return {'Error': 'Input parameters doesnot contain client_id or url'}

    data = {}
    ret_val = {}
    data["client"] = fl_server_update_invoke_obj.get_data_from_server(client_id)
    data["server"] = fl_server_update_invoke_obj.get_weights_from_server()
    data["train_images_url"] = params["train_images_url"]
    data["train_labels_url"] = params["train_labels_url"]
    data["train_labels_url"] = params["train_labels_url"]
    data["test_images_url"] = params["test_images_url"]
    data["test_labels_url"] = params["test_labels_url"]
    data["data_sampling"] = params["data_sampling"]
    data["model"] = params["model"]


    data["lr"] = params["lr"]
    data["optim"] = params["optim"]
    data["local_epochs"] = params["local_epochs"]

    data = bson.BSON.encode(data)
    if "cloud" in client_type:
        try:
            client_response = requests.post(url,
                                           allow_redirects=False, data=data,
                                           proxies=fl_server_update_invoke_obj.proxies)
            client_new_weights = client_response.content
            client_response.raise_for_status()
        except requests.exceptions.HTTPError as httpErr:
            ret_val['Error'] = url
            return ret_val
        except requests.exceptions.ConnectionError as connErr: 
            ret_val['Error'] = connErr
            return ret_val
        except requests.exceptions.Timeout as timeOutErr:
            ret_val['Error'] = timeOutErr  
            return ret_val
        except requests.exceptions.RequestException as reqErr:
            ret_val['Error'] = reqErr  
            return ret_val
    elif "openwhisk" in client_type:
        data = dumps(data)
        data = json.loads(data)
        try:
            client_response = requests.post(url, allow_redirects=False, json=data, verify=False)
            client_new_weights = client_response.content
            client_response.raise_for_status()
        except requests.exceptions.HTTPError as httpErr:
            ret_val['Error'] = url
            return ret_val
        except requests.exceptions.ConnectionError as connErr: 
            ret_val['Error'] = connErr
            return ret_val
        except requests.exceptions.Timeout as timeOutErr:
            ret_val['Error'] = timeOutErr  
            return ret_val
        except requests.exceptions.RequestException as reqErr:
            ret_val['Error'] = reqErr  
            return ret_val
        #client_new_weights = requests.post(url,
                                        #    allow_redirects=True, json=data, verify=False, timeout=600).content
    else:
        try: 
            client_response = requests.post(url, allow_redirects=False, data=data, verify=False)
            client_new_weights = client_response.content
            client_response.raise_for_status()
        except requests.exceptions.HTTPError as httpErr:
            ret_val['Error'] = httpErr
            return ret_val
        except requests.exceptions.ConnectionError as connErr: 
            ret_val['Error'] = connErr
            return ret_val
        except requests.exceptions.Timeout as timeOutErr:
            ret_val['Error'] = timeOutErr  
            return ret_val
        except requests.exceptions.RequestException as reqErr:
            ret_val['Error'] = reqErr  
            return ret_val
        # client_new_weights = requests.post(url,
        #                                    allow_redirects=True, data=data, verify=False, timeout=600).content

    client_new_weights = loads(client_new_weights)

    client_new_weights = bson.BSON(client_new_weights).decode()

    fl_server_update_invoke_obj.write_updated_weights_client(client_new_weights["weights"],
                                                             client_new_weights["cardinality"],
                                                             "client_" + str(client_id))

    
    ret_val['result'] = "executed_Client_" + str(client_id)
    # ret_val['result'] = "executed_Client_" + str(data["lr"])
    return ret_val
