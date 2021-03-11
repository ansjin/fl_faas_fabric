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

functions_meta = []
from commons.Logger import ScriptLogger
logging.basicConfig(level=logging.DEBUG)
logger = ScriptLogger(__name__, 'SWI.log')
logger.setLevel(logging.DEBUG)
logging.captureWarnings(True)


async def deploy_to_clusters(configfile: str, provider: str, scenario_name: str, functions_list: list,
                             cluster_obj: BaseDeployment = None,
                       providers_list: list = None, all_clusters: bool = False):
    with open(configfile, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            if all_clusters:
                for cluster in data['providers'][provider]:
                    curr_cluster = data['providers'][provider][cluster]
                    scenario = data['scenarios'][scenario_name]

                    for function in functions_list:
                        function_object = scenario['functions'][function][provider]
                        await cluster_obj.deploy(curr_cluster, function, function_object)

            else:
                for cluster_name in providers_list:
                    for cluster in data['providers'][provider]:
                        curr_cluster = data['providers'][provider][cluster]
                        scenario = data['scenarios'][scenario_name]
                        if cluster_name == cluster:
                            for function in functions_list:
                                function_object = scenario['functions'][function][provider]
                                await cluster_obj.deploy(curr_cluster, function, function_object)
                            break
        except yaml.YAMLError as exc:
            print(exc)


async def remove_from_clusters(configfile: str, provider: str, scenario_name: str,
                               functions_list: list, cluster_obj: BaseDeployment = None,
                               providers_list: list = None, all_clusters: bool = False):
    with open(configfile, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            if all_clusters:
                for cluster in data['providers'][provider]:
                    curr_cluster = data['providers'][provider][cluster]
                    scenario = data['scenarios'][scenario_name]

                    for function in functions_list:
                        function_object = scenario['functions'][function][provider]
                        await cluster_obj.delete(curr_cluster, function, function_object)

            else:
                for cluster_name in providers_list:
                    for cluster in data['providers'][provider]:
                        curr_cluster = data['providers'][provider][cluster]
                        scenario = data['scenarios'][scenario_name]
                        if cluster_name == cluster:
                            for function in functions_list:
                                function_object = scenario['functions'][function][provider]
                                await cluster_obj.delete(curr_cluster, function, function_object)
                            break
        except yaml.YAMLError as exc:
            print(exc)


async def main(argv):
    openwhisk_obj = OpenWhiskDeployment()
    google_obj = GoogleDeployment()
    configfile = ''
    all_providers = False
    ow_providers_list = []
    gcf_providers_list = []
    functions_list = []
    scenario_name = ""
    deployment = False
    remove = False
    meta = False

    try:
        arguments, values = getopt.getopt(argv, "hc:ao:g:s:f:drm", ["help", "configfile=", "all_providers",
                                                                 "ow_providers_list=", "gcf_providers_list=",
                                                                "scenario_name=", "functions_list=",
                                                                 "deploy", "remove", "get_meta_data"])
    except getopt.GetoptError:
        print('main.py -c <configfile path> -a <for all providers> '
              '-o <OW provider_list separated by comma> -g <GCF provider_list separated by comma>  '
              '-s <scenario_name> -f <functions list separated by comma> '
              '-m <for saving functions meta data in a file>'
              '-d <for deploying> -r <for removing>')
        sys.exit(2)

    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print('python3 deploy.py \n -c <configfile path> \n -a <for all providers> '
                  '\n -o <OW provider_list separated by comma> \n -g <GCF provider_list separated by comma>'
                  '\n -s <scenario_name> '
                  '\n -f <functions  separated by comma> \n -m <for saving functions meta data in a file> '
                  '\n -d <for deploying> \n -r <for removing>')
        elif current_argument in ("-c", "--configfile"):
            configfile = current_value
        elif current_argument in ("-a", "--all_providers"):
            all_providers = True
        elif current_argument in ("-d", "--deploy"):
            deployment = True
        elif current_argument in ("-r", "--remove"):
            remove = True
        elif current_argument in ("-o", "--ow_providers_list"):
            all_arguments = current_value.split(',')
            ow_providers_list = all_arguments
        elif current_argument in ("-g", "--gcf_providers_list"):
            all_arguments = current_value.split(',')
            gcf_providers_list = all_arguments

        elif current_argument in ("-s", "--scenario_name"):
            scenario_name = current_value

        elif current_argument in ("-f", "--functions_list"):
            all_arguments = current_value.split(',')
            functions_list = all_arguments

    tasks: List[asyncio.Task] = []

    if deployment:
        tasks.append(
            asyncio.create_task(
                deploy_to_clusters(configfile, 'openwhisk', scenario_name,
                                   functions_list, openwhisk_obj, ow_providers_list, all_providers)
            )
        )
        tasks.append(
            asyncio.create_task(
                deploy_to_clusters(configfile, 'google', scenario_name,
                                   functions_list, google_obj, gcf_providers_list, all_providers)
            )
        )
    elif remove:
        tasks.append(
            asyncio.create_task(
                remove_from_clusters(configfile, 'openwhisk', scenario_name,
                                     functions_list, openwhisk_obj, ow_providers_list, all_providers)
            )
        )
        tasks.append(
            asyncio.create_task(
                remove_from_clusters(configfile, 'google', scenario_name,
                                     functions_list, google_obj, gcf_providers_list, all_providers)
            )
        )

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
