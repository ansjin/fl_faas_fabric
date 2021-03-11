#!/usr/bin/env python
import subprocess
import sys
import os
sys.path.append(os.path.abspath('../'))
from Clusters import BaseDeployment
import logging
import json
import yaml

from commons.Logger import ScriptLogger
logging.basicConfig(level=logging.DEBUG)
logger = ScriptLogger(__name__, 'SWI.log')
logger.setLevel(logging.DEBUG)
logging.captureWarnings(True)

class GoogleDeployment(BaseDeployment):

    def authentication(self, config__auth_object: object) -> None:
        """ Asynchronous function which creates authentication
            Args:
                config__auth_object:
                    Object - All required authentication information

            Returns:
                None
        """
        # Step 1: Export authorization
        with open('./Clusters/Google/config.json', 'w') as f:
            json.dump(config__auth_object, f, indent=2)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
            '/Users/anshuljindal/Documents/TUM/fl_faas_fabric/Clusters/Google/config.json'

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
        process = subprocess.Popen(
            ["gcloud", "functions", "deploy", func_name,
             "--region", fun_object["region"],
             "--entry-point", fun_object["entry-point"],
             "--memory", fun_object["memory"],
             "--source", fun_object["func_path"],
             "--runtime", fun_object["runtime"],
             "--timeout", fun_object["timeout"],
             "--project", config_object["auth"]["project_id"],
             "--trigger-http",
             "--allow-unauthenticated"], stdout=subprocess.PIPE)
        while True:
            line = process.stdout.readline()
            if not line:
                break
            logger.debug(line.decode('utf-8').split("\n"))

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

        # Step 2: Final Deploy
        process = subprocess.Popen(
            ["gcloud", "functions", "delete", func_name,
             "--region", fun_object["region"],
             "--project", config_object["auth"]["project_id"], '--quiet'], stdout=subprocess.PIPE)
        while True:
            line = process.stdout.readline()
            if not line:
                break
            logger.debug(line.decode('utf-8').split("\n"))

        return "Deleted"
