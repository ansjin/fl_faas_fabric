#!/bin/sh

echo "Building ansjin/fl_server:armhf_client..."
docker build -t ansjin/fl_server:armhf_client . -f Dockerfiles.armhf
docker push ansjin/fl_server:armhf_client

