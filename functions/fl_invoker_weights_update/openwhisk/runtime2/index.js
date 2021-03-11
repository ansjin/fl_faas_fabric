'use strict';
let MongoClient = require('mongodb').MongoClient;
const request = require('request');
const Q = require('q');
const BSON = require('bson');
process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = 0;

let proxies = {
    "http": "http://proxy.in.tum.de:8080/",
    "https": "http://proxy.in.tum.de:8080/",
    "ftp": "ftp://proxy.in.tum.de:8080/",
    "no_proxy": "172.24.65.16"
};

function get_data_from_server(client_id_num, mongo_url, mongo_db, collection_name) {
    var deferred = Q.defer();
    let data_client_id = "data_client_" + client_id_num;
    MongoClient.connect(mongo_url, {useUnifiedTopology: true}, function (err, db) {
        if (err) {
            //console.log(err);
            deferred.resolve(false); // username exists

        }
        var dbo = db.db(mongo_db);
        dbo.collection(collection_name).find({'key': data_client_id}).toArray(function (err, result) {
            if (err) return err;
            result = result[0]["data"];
            db.close();
            deferred.resolve(result);
        });
    });
    return deferred.promise;
}

function get_weights_from_server(client_id_num, mongo_url, mongo_db, collection_name) {
    var deferred = Q.defer();
    MongoClient.connect(mongo_url, function (err, db) {
        if (err) {
            //console.log(err);
            deferred.resolve(false); // username exists

        }
        var dbo = db.db(mongo_db);
        dbo.collection(collection_name).find({'key': "Server"}).toArray(function (err, result) {
            if (err) {
            //console.log(err);
            deferred.resolve(false); // username exists

        }
            result = result[0]["weights"];
            db.close();
            deferred.resolve(result);
        });
    });
    return deferred.promise;
}

function write_updated_weights_client(client_id_num, mongo_url, mongo_db, collection_name, weights_serialized, cardinality) {
    var deferred = Q.defer();
    let key = "client_" + client_id_num;
    var filter = {'key': key};
    MongoClient.connect(mongo_url, function (err, db) {
        if (err) {
            //console.log(err);
            deferred.resolve(false); // username exists

        }
        var dbo = db.db(mongo_db);
        dbo.collection(collection_name).find(filter).toArray(function (err, result) {
            if (err) {
                let values = {'key': key, 'weights': weights_serialized, 'cardinality': parseInt(cardinality)};
                dbo.collection(collection_name).insertOne(values, function (err, result2) {
                    if (err) {
                        //console.log(err);
                        deferred.resolve(false); // username exists

                    }
                    console.log("1 document inserted");
                    db.close();
                    deferred.resolve(result);
                });

            } else {

                var new_values = {'$set': {'weights': weights_serialized, 'cardinality': parseInt(cardinality)}};
                dbo.collection(collection_name).updateOne(filter, new_values, function (err, res) {
                    if (err) {
                        //console.log(err);
                        deferred.resolve(false); // username exists

                    }
                    console.log("1 document updated");
                    db.close();
                    deferred.resolve(result);
                });
            }

        });
    });
    return deferred.promise;
}
function main(params) {
    var deferred = Q.defer();
    if (params) {
        let client_id = params["client_id"];
        let url = params["url"];
        let client_type = params["client_type"]

        let mongo_url = "mongodb://" + params["mongo"]["url"];
        let mongo_db = params["mongo"]["db"];
        let collection_name = params["mongo"]["collection"];
        let data = {};
        let ret_val = {};
        ret_val['result'] = "executed_Client_" + client_id;
        let err_ret_val = {};
        err_ret_val['result'] = "err_Client_" + client_id;
        get_data_from_server(client_id, mongo_url, mongo_db, collection_name).then(function (client_data) {
            data["client"] = client_data;
            get_weights_from_server(client_id, mongo_url, mongo_db, collection_name).then(function (server_data) {
                data["server"] = server_data;
                data["train_images_url"] = params["train_images_url"];
                data["train_labels_url"] = params["train_labels_url"];
                data["train_labels_url"] = params["train_labels_url"];
                data["test_images_url"] = params["test_images_url"];
                data["test_labels_url"] = params["test_labels_url"];
                data["data_sampling"] = params["data_sampling"];
                data["model"] = params["model"];
                let data1 = BSON.serialize(data);
                //console.log(data);
                if (client_type.includes("cloud")) {
                    let proxiedRequest = request.defaults({'proxy': "http://proxy.in.tum.de:8080/"});
                    proxiedRequest.post(url, {
                        body: data1,
                        headers: {"content-type": 'application/octet-stream'},
                    }, (error, res, body) => {
                        if (error) {
                            console.error(error);
                            deferred.resolve(err_ret_val);
                        }
                        //console.log(`statusCode: ${res.statusCode}`);
                        //console.log(body);
                        body = JSON.parse(body);
                        body = BSON.deserialize(body);
                        write_updated_weights_client(client_id, mongo_url, mongo_db, collection_name,
                            body["weights"], body["cardinality"]).then(function (server_data){
                           deferred.resolve(ret_val);
                        })

                    })
                } else if (client_type.includes("openwhisk")) {
                    request.post(url, {
                        json: JSON.parse(JSON.stringify(data1)),
                    }, (error, res, body) => {
                        if (error) {
                            console.error(error);
                            deferred.resolve(err_ret_val);
                        }
                        //console.log(`statusCode: ${res.statusCode}`);
                        //console.log(body);

                        write_updated_weights_client(client_id, mongo_url, mongo_db, collection_name,
                            body["weights"], body["cardinality"]).then(function (server_data){
                            deferred.resolve(ret_val);
                        })

                    })

                } else {
                    request.post(url, {
                        body: data1,
                        headers: {"content-type": 'application/octet-stream'}
                    }, (error, res, body) => {
                        if (error) {
                            console.error(error);
                            deferred.resolve(err_ret_val);
                        }
                        //console.log(`statusCode: ${res.statusCode}`);
                        //console.log(body);
                        body = JSON.parse(body);
                        body = BSON.deserialize(body);

                        write_updated_weights_client(client_id, mongo_url, mongo_db, collection_name,
                            body["weights"], body["cardinality"]).then(function (server_data){
                                deferred.resolve(ret_val);

                        });

                    })

                }
            });
        });
    }
    return deferred.promise;
}
module.exports.main = main;
