#!/usr/bin/env python
import yaml
import sys, getopt
from typing import List
import traceback
import asyncio
import json
import logging

from Clusters import BaseDeployment
from Clusters import OpenWhiskDeployment
from Clusters import GoogleDeployment
from FederatedLearning import FederatedLearning

functions_meta = []
from commons.Logger import ScriptLogger
logging.basicConfig(level=logging.DEBUG)
logger = ScriptLogger(__name__, 'SWI.log')
logger.setLevel(logging.DEBUG)
logging.captureWarnings(True)


async def clean_up_setup(configfile: str, fl_learning_obj: FederatedLearning,
                         openwhisk_obj: OpenWhiskDeployment, gcf_obj: GoogleDeployment):
    with open(configfile, 'r') as stream:
        try:
            data = yaml.safe_load(stream)

            # step1: Firstly, remove previous clients and manager functions if they exist
            await fl_learning_obj.remove_managing_functions(configfile, openwhisk_obj,
                                                            data['scenarios']['federated_learning'][
                                                                'clients_info']['total_num_clients'],
                                                            'openwhisk',
                                                            "invasic_cluster",
                                                            )

            # step2: Remove openwhisk clients
            await fl_learning_obj.remove_clients(
                data['scenarios']['federated_learning']['clients_info']['edge_clients_count'],
                data['scenarios']['federated_learning']['clients_info']['edge_clients_count'] +
                data['scenarios']['federated_learning']['clients_info']['ow_clients_count'],
                configfile, openwhisk_obj, 'openwhisk', "invasic_cluster")

            # step3: Remove GCF clients
            await fl_learning_obj.remove_clients(
                data['scenarios']['federated_learning']['clients_info']['edge_clients_count'] +
                data['scenarios']['federated_learning']['clients_info']['ow_clients_count'],
                data['scenarios']['federated_learning']['clients_info']['total_num_clients'],
                configfile, gcf_obj, 'google', "gcf_cluster")

        except yaml.YAMLError as exc:
            print(exc)


async def create_setup(configfile: str, fl_learning_obj: FederatedLearning,
                         openwhisk_obj: OpenWhiskDeployment, gcf_obj: GoogleDeployment):
    with open(configfile, 'r') as stream:
        try:
            data = yaml.safe_load(stream)

            # step1: Firstly, create parameters files
            await fl_learning_obj.create_all_params_files(configfile)

            # step2: Deploy managing functions
            await fl_learning_obj.deploy_managing_functions(configfile, openwhisk_obj,
                                                            data['scenarios']['federated_learning'][
                                                                'clients_info']['total_num_clients'],
                                                            'openwhisk',
                                                            "invasic_cluster",
                                                            )

            # step2: Deploy openwhisk clients
            await fl_learning_obj.deploy_clients(
                data['scenarios']['federated_learning']['clients_info']['edge_clients_count'],
                data['scenarios']['federated_learning']['clients_info']['edge_clients_count'] +
                data['scenarios']['federated_learning']['clients_info']['ow_clients_count'],
                configfile, openwhisk_obj, 'openwhisk', "invasic_cluster")

            # step3: Deploy GCF clients
            await fl_learning_obj.deploy_clients(
                data['scenarios']['federated_learning']['clients_info']['edge_clients_count'] +
                data['scenarios']['federated_learning']['clients_info']['ow_clients_count'],
                data['scenarios']['federated_learning']['clients_info']['total_num_clients'],
                configfile, gcf_obj, 'google', "gcf_cluster")

        except yaml.YAMLError as exc:
            print(exc)


async def main(argv):
    openwhisk_obj = OpenWhiskDeployment()
    google_obj = GoogleDeployment()
    fl_learning_obj = FederatedLearning()
    configfile = ''
    deployment = False
    remove = False
    start_fl = False

    try:
        arguments, values = getopt.getopt(argv, "hc:drs", ["help", "configfile=", "deploy", "remove", "start_fl"])
    except getopt.GetoptError:
        print('main.py -c <configfile path> '
              '-d <for deploying> -r <for removing>')
        sys.exit(2)

    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print('python3 deploy.py \n -c <configfile path>'
                  '\n -d <for deploying> \n -r <for removing> \n -s <for starting federated learning>')
        elif current_argument in ("-c", "--configfile"):
            configfile = current_value
        elif current_argument in ("-d", "--deploy"):
            deployment = True
        elif current_argument in ("-r", "--remove"):
            remove = True
        elif current_argument in ("-s", "--start_fl"):
            start_fl = True

    tasks: List[asyncio.Task] = []

    if deployment:
        tasks.append(
            asyncio.create_task(
                create_setup(configfile, fl_learning_obj, openwhisk_obj, google_obj)
            )
        )
    elif remove:
        tasks.append(
            asyncio.create_task(
                clean_up_setup(configfile, fl_learning_obj, openwhisk_obj, google_obj)
            )
        )
    elif start_fl:
        fl_learning_obj.start_fl_learning(configfile, openwhisk_obj)

    # wait for all workers
    if len(tasks):
        try:
            await asyncio.wait(tasks)
        except Exception as e:
            print("Exception in main worker loop")
            print(e)
            traceback.print_exc()

        print("All deployment/removal finished")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
