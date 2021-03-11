#!/usr/bin/env python
import subprocess
import sys
import os
sys.path.append(os.path.abspath('../'))
from Clusters import BaseDeployment
import logging
import yaml
import time
from pandas import DataFrame

from commons.Logger import ScriptLogger
logging.basicConfig(level=logging.DEBUG)
logger = ScriptLogger(__name__, 'SWI.log')
logger.setLevel(logging.DEBUG)
logging.captureWarnings(True)


class OpenWhiskDeployment(BaseDeployment):

    def authentication(self, config__auth_object: object) -> None:
        """ Asynchronous function which creates authentication
            Args:
                config__auth_object:
                    Object - All required authentication information

            Returns:
                None
        """
        # Step 1: Export authorization
        os.environ["OW_AUTH"] = config__auth_object["ow_auth"]
        os.environ["OW_APIHOST"] = config__auth_object["ow_api_host"]
        os.environ["OW_APIGW_ACCESS_TOKEN"] = config__auth_object["ow_apigw_access_token"]

    async def deploy(self, config_object: object, func_name: str, fun_object: object) -> str:
        """ Asynchronous function which creates the deployment of the serverless function using gcloud cli
            Args:
                config_object:
                    Object - All required authentication information

                func_name:
                    String - Name of the function

                fun_object:
                    Object - All required function information
            Returns:
                String
        """
        # Step 1: Export authorization
        self.authentication(config_object["auth"])

        # Step 2: Final Deploy
        if fun_object["docker_image"]:
            process = subprocess.Popen(["wsk", "action", "create", func_name,
                                        fun_object["func_path"],
                                        "--docker", fun_object['docker_image'],
                                        "--memory", fun_object['memory'],
                                        "--timeout", fun_object['timeout'],
                                        "--concurrency", fun_object['concurrency'],
                                        "--web", "raw", "-i"], stdout=subprocess.PIPE)
        else:
            process = subprocess.Popen(["wsk", "action", "create", func_name,
                                        fun_object["func_path"],
                                        "--kind", fun_object["runtime"],
                                        "--memory", fun_object['memory'],
                                        "--timeout", fun_object['timeout'],
                                        "--concurrency", fun_object['concurrency'],
                                        "--web", "raw", "-i"], stdout=subprocess.PIPE)

        output, error_remove = process.communicate()
        for line in output.decode().split("\n"):
            logger.debug(line)
        return "Deployed"

    async def delete(self, config_object: object, func_name: str, fun_object: object) -> str:
        """ Asynchronous function which creates the deployment of the serverless function using gcloud cli
            Args:
                config_object:
                    Object - All required authentication information

                func_name:
                    String - Name of the function

                fun_object:
                    Object - All required function information
            Returns:
                String
        """
        # Step 1: Export authorization
        self.authentication(config_object["auth"])

        # Step 2: Final delete

        process = subprocess.Popen(["wsk", "action", "delete", func_name, "-i"], stdout=subprocess.PIPE)

        output, error_remove = process.communicate()
        for line in output.decode().split("\n"):
            logger.debug(line)
        return "Deleted"

    def invoke_function(self, function_name: str, param_file: str = None) -> list:
        metrics = []
        if param_file:
            process = subprocess.Popen(["wsk", "action", "invoke", function_name, "-P", param_file, "-i", "-r"],
                                       stdout=subprocess.PIPE)
        else:
            process = subprocess.Popen(["wsk", "action", "invoke", function_name, "-i", "-r"],
                                       stdout=subprocess.PIPE)
        while True:

            line = process.stdout.readline()
            # print(line.decode('utf-8'))
            if "accuracy" in line.decode('utf-8'):
                acc = line.decode('utf-8').strip().replace(',', '').split("\"accuracy\": ")
                # print(acc)
                metrics.append(acc[1])
            elif "loss" in line.decode('utf-8'):
                loss = line.decode('utf-8').strip().replace(',', '').split("\"loss\": ")
                # print(loss)
                metrics.append(loss[1])
            if not line:
                break
            # logger.info(line.decode('utf-8'))

        return metrics

    def invoke_function_nb(self, function_name: str, param_file: str = None) -> str:
        "Asssuming one function called per invocation"
        time.sleep(0.05)
        activation_id = ""
        print("wsk action invoke ", function_name, "-P ", param_file, "-i")
        if param_file:
            process = subprocess.Popen(["wsk", "action", "invoke", function_name, "-P", param_file, "-i"],
                                       stdout=subprocess.PIPE)
        else:
            process = subprocess.Popen(["wsk", "action", "invoke", function_name, "-i"], stdout=subprocess.PIPE)
        while True:
            line = process.stdout.readline()
            if "id" in line.decode('utf-8'):
                activation_id = line.decode('utf-8').strip().split("id ")
                activation_id = activation_id[1]
            if not line:
                    break
        return activation_id

    def check_results_activation_ids(self,  config_object: object, count: int,
                                           activation_ids: list, client_activation_map: dict,
                                           activations_df: DataFrame):
        activation_count = count
        self.authentication(config_object["auth"])

        for id in activation_ids:
            process = subprocess.Popen(["wsk", "activation", "result", "-i", id],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            while True:
                line_stdout = process.stdout.readline()
                line_stderr = process.stderr.readline()
                if "Client" in line_stdout.decode('utf-8'):
                    activation_count = activation_count + 1
                    activation_ids.remove(id)
                    activations_df.loc[activations_df['activation_id'] == id, ['end_time']] = time.time()
                    break
                elif "Failed" in line_stdout.decode('utf-8'):
                    #print(line_stdout.decode('utf-8'))
                    client_failed = client_activation_map[id]
                    client_activation_map.pop(id)
                    #print("Failed for Client No: {0}".format(client_failed))
                    activation_ids.remove(id)
                    activations_df.loc[activations_df['activation_id'] == id, ['end_time']] = time.time()

                    time.sleep(0.12)
                    new_activation_id = self.invoke_function_nb("fl_invoker_weights_update" + str(client_failed),
                                                    "params/clients/" + "client" + str(client_failed) + ".json")

                    activations_df = activations_df.append(
                        {"client_id": client_failed, "activation_id": new_activation_id, "start_time": time.time(),
                         "end_time": 0}, ignore_index=True)

                    activation_ids.append(new_activation_id)
                    client_activation_map[new_activation_id] = client_failed
                elif "Internal" in line_stdout.decode('utf-8'):
                    #print(line_stdout.decode('utf-8'))
                    client_failed = client_activation_map[id]
                    client_activation_map.pop(id)
                    #print("Failed for Client No: {0}".format(client_failed))
                    activation_ids.remove(id)
                    activations_df.loc[activations_df['activation_id'] == id, ['end_time']] = time.time()

                    time.sleep(0.12)
                    new_activation_id = self.invoke_function_nb("fl_invoker_weights_update" + str(client_failed),
                                                                "params/clients/" + "client" + str(
                                                                    client_failed) + ".json")
                    activation_ids.append(new_activation_id)

                    activations_df = activations_df.append(
                        {"client_id": client_failed, "activation_id": new_activation_id, "start_time": time.time(),
                         "end_time": 0}, ignore_index=True)


                    client_activation_map[new_activation_id] = client_failed
                elif "Client" not in line_stdout.decode('utf-8') and len(line_stdout.decode('utf-8')) > 0 and "{" not in line_stdout.decode('utf-8')\
                        and "}" not in line_stdout.decode('utf-8'):
                    #print(line_stdout.decode('utf-8'))
                    client_failed = client_activation_map[id]
                    client_activation_map.pop(id)
                    #print("Failed for Client No: {0}".format(client_failed))
                    activation_ids.remove(id)
                    activations_df.loc[activations_df['activation_id'] == id, ['end_time']] = time.time()

                    time.sleep(0.12)
                    new_activation_id = self.invoke_function_nb("fl_invoker_weights_update" + str(client_failed),
                                                                "params/clients/" + "client" + str(
                                                                    client_failed) + ".json")
                    activation_ids.append(new_activation_id)

                    activations_df = activations_df.append(
                        {"client_id": client_failed, "activation_id": new_activation_id, "start_time": time.time(),
                         "end_time": 0}, ignore_index=True)

                    client_activation_map[new_activation_id] = client_failed
                # elif "resource" not in line_stderr.decode('utf-8') and "Client" not in line_stdout.decode('utf-8'):
                #     print(line_stderr.decode('utf-8'))
                #     print(line_stdout.decode('utf-8'))
                if not line_stdout:
                    break
        return activation_count, activation_ids, client_activation_map, activations_df

    def check_if_all_activations_completed(self, config_object: object,
                                                 activation_ids: list, client_activation_map: dict,
                                                 activations_df: DataFrame,
                                                 interval: int = 1) -> DataFrame:
        count_activations = len(activation_ids)
        count = 0
        while count != count_activations:
            time.sleep(interval)
            count, activation_ids, client_activation_map, activations_df = \
                self.check_results_activation_ids(config_object,
                                                  count, activation_ids, client_activation_map, activations_df)

        return activations_df
