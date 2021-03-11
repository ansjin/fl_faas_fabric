#!/usr/bin/env python
import yaml
import sys, getopt
from typing import List
import traceback
import asyncio
import json
import os
import shutil
import time
import logging
import numpy as np
import pandas as pd
from Clusters import BaseDeployment
from Clusters import OpenWhiskDeployment
from Clusters import GoogleDeployment

functions_meta = []
from commons.Logger import ScriptLogger
logging.basicConfig(level=logging.DEBUG)
logger = ScriptLogger(__name__, 'SWI.log')
logger.setLevel(logging.DEBUG)
logging.captureWarnings(True)


class FederatedLearning:
    async def deploy_to_clusters(self, configfile: str, provider: str, curr_cluster: str, base_function_name,
                                 func_names, cluster_obj: BaseDeployment = None):
        with open(configfile, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                curr_cluster = data['providers'][provider][curr_cluster]
                scenario = data['scenarios']['federated_learning']
                function_object = scenario['functions'][base_function_name][provider]

                for function in func_names:
                    await cluster_obj.deploy(curr_cluster, function, function_object)

            except yaml.YAMLError as exc:
                print(exc)

    async def remove_from_clusters(self, configfile: str, provider: str, curr_cluster: str, base_function_name,
                                 func_names, cluster_obj: BaseDeployment = None):
        with open(configfile, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                curr_cluster = data['providers'][provider][curr_cluster]
                scenario = data['scenarios']['federated_learning']
                function_object = scenario['functions'][base_function_name][provider]

                for function in func_names:
                    await cluster_obj.delete(curr_cluster, function, function_object)

            except yaml.YAMLError as exc:
                print(exc)

    async def create_clients_param_files(self, start_num, num_clients, base_url, clients_param_info_object, client_type,
                                   path_to_client_function):
        client_ids = [str(i) for i in range(start_num, num_clients + start_num)]

        urls = [base_url + "client" + str(i) for i in client_ids]
        for i in range(num_clients):
            data = {}
            data["client_id"] = client_ids[i]
            data["client_type"] = client_type

            data["url"] = urls[i]
            data["train_images_url"] = clients_param_info_object["training_test_data"]["train_images_url"]
            data["train_labels_url"] = clients_param_info_object["training_test_data"]["train_labels_url"]
            data["test_images_url"] = clients_param_info_object["training_test_data"]["test_images_url"]
            data["test_labels_url"] = clients_param_info_object["training_test_data"]["test_labels_url"]
            data["mongo"] = clients_param_info_object["mongo_config"]
            data["data_sampling"] = clients_param_info_object["data_sampling"]
            data["model"] = clients_param_info_object["data_sampling"]["model"]
            data["lr"] = clients_param_info_object["model_params"]["lr"]
            data["optim"] = clients_param_info_object["model_params"]["optim"]
            data["local_epochs"] = clients_param_info_object["model_params"]["local_epochs"]
            filename = "client" + str(client_ids[i]) + ".json"
            path_to_write = os.path.join(path_to_client_function, filename)
            # print(path_to_write)
            with open(path_to_write, 'w') as outfile:
                json.dump(data, outfile, indent=4)

    async def create_fl_server_init_param_file(self, num_clients, clients_param_info_object, path_to_init_function):
        data = {
            "num_clients": num_clients,
            "train_images_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "train_labels_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "test_images_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "test_labels_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "mongo": clients_param_info_object["mongo_config"],
            "data_sampling": clients_param_info_object["data_sampling"]
        }
        with open(path_to_init_function, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    async def create_usage_num_clients_param_file(self, num_clients, clients_param_info_object, path_to_init_function):
        client_ids = ["client_" + str(i) for i in range(0, num_clients + 0)]
        data = {
            "num_clients": num_clients,
            "train_images_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "train_labels_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "test_images_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "test_labels_url": clients_param_info_object["training_test_data"]["train_images_url"],
            "mongo": clients_param_info_object["mongo_config"],
            "data_sampling": clients_param_info_object["data_sampling"],
            "client_round_ids": client_ids
        }
        with open(path_to_init_function, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    async def create_all_params_files(self, configfile):
        tasks: List[asyncio.Task] = []
        with open(configfile, 'r') as stream:
            try:
                data = yaml.safe_load(stream)

                file_path = 'params'
                try:
                    shutil.rmtree(file_path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                os.mkdir('./params')
                os.mkdir('./params/clients')


                tasks.append(
                    asyncio.create_task(
                        self.create_fl_server_init_param_file(
                            data['scenarios']['federated_learning']['clients_info']['total_num_clients'],
                            data['scenarios']['federated_learning'],
                            data['scenarios']['federated_learning']['functions']
                            ['fl_server_init']['openwhisk']['params_file_path']
                        )
                    )
                )
                tasks.append(
                    asyncio.create_task(
                        self.create_usage_num_clients_param_file(
                            data['scenarios']['federated_learning']['clients_info']['total_num_clients'],
                            data['scenarios']['federated_learning'],
                            data['scenarios']['federated_learning']['functions']
                            ['fl_server_aggregator']['openwhisk']['params_file_path']
                        )
                    )
                )
                tasks.append(
                    asyncio.create_task(
                        self.create_clients_param_files(
                            data['scenarios']['federated_learning']['clients_info']['edge_clients_count'],
                            data['scenarios']['federated_learning']['clients_info']['ow_clients_count'],
                            data['providers']['openwhisk']['invasic_cluster']['base_url'],
                            data['scenarios']['federated_learning'],
                            "openwhisk_local",
                            data['scenarios']['federated_learning'
                                              '']['functions']['fl_client']['openwhisk']['params_file_path'])
                    )
                )
                tasks.append(
                    asyncio.create_task(
                        self.create_clients_param_files(
                            data['scenarios']['federated_learning']['clients_info']['edge_clients_count'] +
                            data['scenarios']['federated_learning']['clients_info']['ow_clients_count'],
                            data['scenarios']['federated_learning']['clients_info']['gcf_clients_count'],
                            data['providers']['google']['gcf_cluster']['base_url'],
                            data['scenarios']['federated_learning'],
                            "cloud",
                            data['scenarios']['federated_learning'
                                              '']['functions']['fl_client']['google']['params_file_path'])
                    )
                )
            except yaml.YAMLError as exc:
                print(exc)

        if len(tasks):
            try:
                await asyncio.wait(tasks)
            except Exception as e:
                print("Exception in main worker loop")
                print(e)
                traceback.print_exc()

            print("All params files created")

    async def deploy_managing_functions(self, configfile, cluster_obj, num_clients, provider_type='openwhisk',
                                        cluster_name='invasic_cluster'):
        tasks: List[asyncio.Task] = []
        tasks.append(
            asyncio.create_task(
                self.deploy_to_clusters(configfile, provider_type, cluster_name, "fl_server_init", ["fl_server_init"],
                                     cluster_obj)
            )
        )
        tasks.append(
            asyncio.create_task(
                self.deploy_to_clusters(configfile, provider_type, cluster_name, "fl_server_aggregator",
                                     ["fl_server_aggregator"],
                                     cluster_obj)
            )
        )
        for client in range(0, num_clients):
            func_name = 'fl_invoker_weights_update' + str(client)
            tasks.append(
                asyncio.create_task(
                    self.deploy_to_clusters(configfile, provider_type, cluster_name, "fl_invoker_weights_update",
                                           [func_name],
                                           cluster_obj)
                )
            )

        if len(tasks):
            try:
                await asyncio.wait(tasks)
            except Exception as e:
                print("Exception in main worker loop")
                print(e)
                traceback.print_exc()

            print("All server managing functions deployed")

    async def remove_managing_functions(self, configfile, cluster_obj, num_clients, provider_type='openwhisk',
                                        cluster_name='invasic_cluster'):
        tasks: List[asyncio.Task] = []
        tasks.append(
            asyncio.create_task(
                self.remove_from_clusters(configfile, provider_type, cluster_name, "fl_server_init", ["fl_server_init"],
                                     cluster_obj)
            )
        )
        tasks.append(
            asyncio.create_task(
                self.remove_from_clusters(configfile, provider_type, cluster_name, "fl_server_aggregator",
                                     ["fl_server_aggregator"],
                                     cluster_obj)
            )
        )
        for client in range(0, num_clients):
            func_name = 'fl_invoker_weights_update' + str(client)
            tasks.append(
                asyncio.create_task(
                    self.remove_from_clusters(configfile, provider_type, cluster_name, "fl_invoker_weights_update",
                                           [func_name],
                                           cluster_obj)
                )
            )
        if len(tasks):
            try:
                await asyncio.wait(tasks)
            except Exception as e:
                print("Exception in main worker loop")
                print(e)
                traceback.print_exc()

            print("All server managing functions removed")

    async def deploy_clients(self, start, end, configfile, cluster_obj, provider_type, cluster_name):
        tasks: List[asyncio.Task] = []
        for client_id in range(start, end):
            tasks.append(
                asyncio.create_task(
                    self.deploy_to_clusters(configfile, provider_type, cluster_name, "fl_client",
                                       ["client_" + str(client_id)],
                                       cluster_obj)
                )
            )
        if len(tasks):
            try:
                await asyncio.wait(tasks)
            except Exception as e:
                print("Exception in main worker loop")
                print(e)
                traceback.print_exc()

            print("All client functions of " + provider_type + " deployed")

    async def remove_clients(self, start, end, configfile, cluster_obj, provider_type, cluster_name):
        tasks: List[asyncio.Task] = []
        for client_id in range(start, end):
            tasks.append(
                asyncio.create_task(
                    self.remove_from_clusters(configfile, provider_type, cluster_name, "fl_client",
                                       ["client_" + str(client_id)],
                                       cluster_obj)
                )
            )
        if len(tasks):
            try:
                await asyncio.wait(tasks)
            except Exception as e:
                print("Exception in main worker loop")
                print(e)
                traceback.print_exc()

            print("All client functions of " + provider_type + " removed")

    def update_usage_num_clients_param_file(self, filename, client_round_ids):
        with open(filename, 'r') as json_file:
            config_data_clients = json_file.read()
        config_clients_data = json.loads(config_data_clients)

        config_clients_data["client_round_ids"] = client_round_ids
        with open(filename, 'w') as outfile:
            json.dump(config_clients_data, outfile, indent=4)

    def start_fl_learning(self, configfile: str, openwhisk_obj: OpenWhiskDeployment):
        client_activation_map = {}
        with open(configfile, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                acc = 0
                iter = 0
                loss = 0

                logger.info(' ############## Experiment : ' + data["name"] + "###################")
                num_clients_per_round = data['scenarios']['federated_learning']['clients_info']['num_clients_per_round']
                total_clients = data['scenarios']['federated_learning']['clients_info'][
                                            'total_num_clients']

                logger.info("Step1: Invoking Initialize function for the weights init")
                fl_server_init_params_path = data['scenarios']['federated_learning']['functions'][
                    'fl_server_init']['openwhisk']['params_file_path']
                metrics = openwhisk_obj.invoke_function("fl_server_init",
                                                        fl_server_init_params_path)


                logger.info("Num Clients per rounds: {0}".format(num_clients_per_round))
                start_time_total = time.time()
                while acc < 0.990:
                    activation_ids = []
                    round_clients = np.random.choice(np.arange(total_clients), num_clients_per_round, replace=False)
                    client_round_ids = ["client_" +  str(i) for i in round_clients]
                    logger.info("Iter: {0}, Clients: {1}".format(iter, client_round_ids))
                    self.update_usage_num_clients_param_file("params/usage_num_clients.json", client_round_ids)

                    activations_df = pd.DataFrame()
                    activations_df["client_id"] = []
                    activations_df["activation_id"] = []
                    activations_df["start_time"] = []
                    activations_df["end_time"] = []

                    logger.info("Starting training, Num Clients: {0}".format(num_clients_per_round))
                    fl_clients_base_path = data['scenarios']['federated_learning']['functions'][
                                        'fl_client']['openwhisk']['params_file_path']

                    start_time_iter = time.time()

                    for index, client in enumerate(round_clients):

                        func_name = 'fl_invoker_weights_update' + str(client)
                        #start_time_fun_process = time.time()
                        # print(func_name)
                        act_start_time = time.time()
                        activation_id = openwhisk_obj.invoke_function_nb(func_name,
                                                                         fl_clients_base_path + 'client' +
                                                                         str(client) + ".json")

                        client_activation_map[activation_id] = str(client)
                        print(activation_id)
                        activation_ids.append(activation_id)
                        activations_df = activations_df.append({"client_id": client,
                                                                "activation_id": activation_id,
                                                                "start_time": act_start_time,
                                                                "end_time": 0}, ignore_index=True)

                    end_invocation_time = time.time()
                    cluster_data = data['providers']['openwhisk']['invasic_cluster']

                    activations_df = openwhisk_obj.check_if_all_activations_completed(
                        cluster_data, activation_ids, client_activation_map, activations_df)

                    activations_df["diff"] = activations_df["end_time"] - activations_df["start_time"]

                    csv_file_name = "logs/csvs/" + data['scenarios']['federated_learning']['model_params'][
                        'local_epochs'] + "_" + data['scenarios']['federated_learning']['data_sampling'][
                        'model'] + "/"
                    activations_df.to_csv(csv_file_name + str(iter) + ".csv")
                    start_time_agg = time.time()
                    fl_server_aggregator_params_path = data['scenarios']['federated_learning']['functions'][
                        'fl_server_aggregator']['openwhisk']['params_file_path']

                    metrics = openwhisk_obj.invoke_function("fl_server_aggregator",
                                                            fl_server_aggregator_params_path)

                    end_time_iter = time.time()
                    acc, loss = float(metrics[0]), float(metrics[1])
                    iter += 1
                    logger.info('Iter: {0}, Loss: {1}, '
                                'Accuracy: {2:.3f}, InvokeTime:{3},'
                                '  aggTime: {4}, total Time: {5}, '
                                'Training Time: {6}'.format(iter,
                                                            loss, acc, (end_invocation_time - start_time_iter),
                                                            (end_time_iter - start_time_agg),
                                                            (end_time_iter - start_time_iter),
                                                            (start_time_agg - end_invocation_time)))
                    acc = round(acc, 3)
                logger.info('TotalIter: {0}, FinalLoss: {1}, '
                            'FinalAccuracy: {2:.3f}, TotalTime: {3}'.format(iter, loss, acc,
                                                                            (time.time() - start_time_total)))
            except yaml.YAMLError as exc:
                print(exc)
