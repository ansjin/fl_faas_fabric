'use strict'
let os = require('os');
let fs = require('fs');
let util = require('util');

function main(params) {
    return new Promise(function(resolve, reject) {
        fs.readFile("/etc/hostname", "utf8", (err, data) => {

            if(err){
                reject({payload:  err})
            }else{
                let val = "";
                val += "Hostname: " + data + "\n";
                val += "Platform: " + os.platform() + "\n";
                val += "Arch: " + os.arch() + "\n";
                val += "CPU count: " + os.cpus().length + "\n";

                val += "Uptime: " + os.uptime() + "\n";

                if (params && params.length && params.indexOf("verbose") > -1) {
                    val += util.inspect(os.cpus()) + "\n";
                    val += util.inspect(os.networkInterfaces())+ "\n";
                }
                resolve({payload:  val})
            }
        });

     });
}
exports.main = main;
