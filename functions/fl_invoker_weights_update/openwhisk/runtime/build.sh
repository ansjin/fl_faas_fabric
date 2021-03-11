#!/bin/sh

echo "Building ansjin/fl_server:invoker..."
docker build -t ansjin/fl_server:invoker .
docker push ansjin/fl_server:invoker

