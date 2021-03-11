/**
 * Responds to any HTTP request.
 *
 * @param {!express:Request} req HTTP request context.
 * @param {!express:Response} res HTTP response context.
 */
'use strict'
let os = require('os');
let fs = require('fs');
let util = require('util');

exports.main = (req, res) => {
        fs.readFile("/etc/hostname", "utf8", (err, data) => {
            if(err){
                res.status(200).send({payload:  err})
            }else{
                let val = "";
                val += "Hostname: " + data + "\n";
                val += "Platform: " + os.platform() + "\n";
                val += "Arch: " + os.arch() + "\n";
                val += "CPU count: " + os.cpus().length + "\n";

                val += "Uptime: " + os.uptime() + "\n";

                if (req && req.length && req.indexOf("verbose") > -1) {
                    val += util.inspect(os.cpus()) + "\n";
                    val += util.inspect(os.networkInterfaces())+ "\n";
                }
                res.status(200).send({payload:  val})
            }
        });
};
