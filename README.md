# FedKeeper [![License: Apache 2](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ansjin/fl_faas_fabric/blob/master/LICENSE)


FedKeeper is  a  client-based  python  tool  for  propagatingFL-client functions over FaaS fabric. 
It’s main objective isto  act  as  a  manager  or  keeper  of  various  client  functions
distributed over different FaaS platforms. It has the following
responsibilities :
- Facilitating the automatic creation, deletion, and invo-cation of FL-client functions 
for each FaaS platform. FedKeeperis integrated with the APIs and SDKs of each FaaS platform 
used in this work.
- Resiliency for FL-client functions.
FedKeeper keeps track of the functions running on each FaaS platform using activation IDs 
and automatically creates or invokes the functions which have stopped or failed 

Read More in our paper: https://dl.acm.org/doi/10.1145/3429880.3430100
<p align="center">
<img src="https://github.com/ansjin/fl_faas_fabric/blob/master/docs/overall_architecture1.png"></img>
</p> 

## Install
1. Firstly, deploy the kubernetes cluster hosting OpenWhisk, you can check this <a href="https://medium.com/@ansjin/openwhisk-deployment-on-a-kubernetes-cluster-7fd3fc2f3726">blog</a>. 
2. Install python packages ``` pip install -r requirements.txt```

## Setup 
1. Clone the repository
2. Rename ```config-sample.yaml``` to ```config.yaml```. 
3. Include the clusters, their authentication, update the base URLs information in it. 
4. Start a Mongo Database and update its configuration in the yaml file. 
5. Do remove the proxies from the functions codes, if not being used.  
For example, below shows the example configuration 
```yaml
version: 0.1
name: test_mnist_nn_100_clients_adam_10_local_epochs_fixed_final_training_time
providers:
  openwhisk:
    invasic_cluster:
      base_url: "https://<ADD OW_AUTH here>@<ADD_IP_HERE>:31001/api/v1/namespaces/guest/actions/"
      auth:
        ow_auth: <ADD OW_AUTH here>
        ow_api_host: <ADD API HOST of OW>
        ow_apigw_access_token: APIGW_ACCESS_TOKEN
      prometheus_config:
        host: ""
        port_openwhisk_metrics: "30325"
        port_node_metrics: "30471"
      configurations:
        num_hosts: 1
        each_host_cores: ""
        each_host_ram: 755gb
        gpu_enabled: false
        cuda_cores: ""
        platform: amd64
        host_names: invasic
        concurrency: "100"
  google:
    gcf_cluster:
      base_url: "https://us-central1-<ADD_PROJECT_ID_HERE>.cloudfunctions.net/"
      auth:
        type: service_account
        project_id: ""
        private_key_id: ""
        private_key: ""
        client_email: ""
        client_id: ""
        auth_uri: ""
        token_uri: ""
        auth_provider_x509_cert_url: ""
        client_x509_cert_url: ""
 scenarios:
  federated_learning:
    server_config:
      server_cluster_name: invasic_cluster
      server_manager_functions:
        - fl_server_init
        - fl_server_aggregator
        - fl_invoker_weights_update

    clients_info:
      total_num_clients: 2
      num_clients_per_round: 2
      edge_clients_count: 0
      ow_clients_count: 1
      gcf_clients_count: 1
      client_functions:
        - fl_client

    data_sampling:
      input_size: 784
      hidden_size: 500
      output_size: 10
      input_shape_x: 28
      input_shape_y: 28
      batch_size: 10
      model: "cnn"
    mongo_config:
      url: <ADD_MONGO_IP_HERE>:27017
      db: flweights
      collection: mnist_weights
    training_test_data:
      train_images_url: "https://storage.googleapis.com/mnist_fl/mnist_train_img.obj"
      train_labels_url: "https://storage.googleapis.com/mnist_fl/mnist_train_labels.obj"
      test_images_url: "https://storage.googleapis.com/mnist_fl/mnist_test_img.obj"
      test_labels_url: "https://storage.googleapis.com/mnist_fl/mnist_test_labels.obj"
    model_params:
      lr: "1e-3"
      optim: "adam"
      local_epochs: "5"

    functions:
      fl_client:
        openwhisk:
          func_path: functions/fl_client/openwhisk/main.py
          docker_image: kkyfury/tensorflow
          memory: "2048"
          timeout: "300000"
          params_file_path: "params/clients/"
          concurrency: "100"

 ```

## Functions

All the functions required for running FL over FaaS fabric are within the functions directory:
- ```fl_client```: For doing the actual training process in a client.
- ```fl_server_init```: For initializing the weights in the mongo database
- ```fl_server_aggregator```: For handling the aggregation of all the updated weights received from the clients. 
- ```fl_invoker_weights_update```: For invoking the clients and sending and receiving the updated weights from them.
 
## Testing the Clusters Deployment
For checking the deployment of the clusters.  
Below are the different parameters
 ```
python3 deploy.py 
 -c <configfile path> 
 -a <for all providers> 
 -o <OW provider_list separated by comma> 
 -g <GCF provider_list separated by comma>
 -s <scenario_name> 
 -f <functions  separated by comma> 
 -d <for deploying> 
 -r <for removing>
 ```

For example, 
1. Deploying Multiple OW and GCF: ``` python3 deploy.py -c ./config.yaml -o invasic_cluster -g gcf_cluster -s test -f nodeinfo -d```
2. For deploying all clusters: ``` python3 deploy.py -c ./config.yaml -a -s test -f nodeinfo -d ```
3. For removing all clusters: ``` python3 deploy.py -c ./config.yaml -a -s test -f nodeinfo -r```
You can use this part for testing whether your clusters are working or not. 

## Starting Federated Learning

1. First, update the ``` config.yaml```  file to have the right parameters and clients for FL. 
2. Secondly, remove the previous deployments if there is any using the command: <br>
``` python3 main.py -c ./config.yaml -r ``` <br>
This will remove all the managing functions and clients from all the FaaS platforms.

2. Create the setup for Federated learning using the command: <br>
``` python3 main.py -c ./config.yaml -d ```<br>
This will create all the managing functions and the clients based on the YAML file in the respective FaaS platforms.

3. Start the Federated learning using the command: <br>
``` python3 main.py -c ./config.yaml -s ```<br>
You can check the logs in the ``` Logs ``` folder


## Reference Our paper:

If you find this code useful in your research, please consider citing:

```
@inproceedings{10.1145/3429880.3430100,
author = {Chadha, Mohak and Jindal, Anshul and Gerndt, Michael},
title = {Towards Federated Learning Using FaaS Fabric},
year = {2020},
isbn = {9781450382045},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3429880.3430100},
doi = {10.1145/3429880.3430100},
abstract = {Federated learning (FL) enables resource-constrained edge devices to learn a shared Machine Learning (ML) or Deep Neural Network (DNN) model, while keeping the training data local and providing privacy, security, and economic benefits. However, building a shared model for heterogeneous devices such as resource-constrained edge and cloud makes the efficient management of FL-clients challenging. Furthermore, with the rapid growth of FL-clients, the scaling of FL training process is also difficult.In this paper, we propose a possible solution to these challenges: federated learning over a combination of connected Function-as-a-Service platforms, i.e., FaaS fabric offering a seamless way of extending FL to heterogeneous devices. Towards this, we present FedKeeper, a tool for efficiently managing FL over FaaS fabric. We demonstrate the functionality of FedKeeper by using three FaaS platforms through an image classification task with a varying number of devices/clients, different stochastic optimizers, and local computations (local epochs).},
booktitle = {Proceedings of the 2020 Sixth International Workshop on Serverless Computing},
pages = {49–54},
numpages = {6},
keywords = {Serverless, Neural networks, Function-as-a-service, Federated learning, FaaS platforms, FaaS},
location = {Delft, Netherlands},
series = {WoSC'20}
}
```
All the data related to paper can be found in ```docs/paper_related_data```

## Help and Contribution

Please add issues if you have a question or found a problem. 

Pull requests are welcome too!

Contributions are most welcome. Please message me if you like the idea and want to contribute. 
