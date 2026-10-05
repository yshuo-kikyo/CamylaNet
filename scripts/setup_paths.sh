#!/bin/bash

export CAMYLANET_ROOT=${CAMYLANET_ROOT:-/data/hdd1/yanshuo}

export camylanet_raw=$CAMYLANET_ROOT/datasets/CamylaNet_raw
export camylanet_preprocessed=$CAMYLANET_ROOT/preprocessed/CamylaNet
export camylanet_results=$CAMYLANET_ROOT/results/CamylaNet

echo "CAMYLANET_ROOT=$CAMYLANET_ROOT"
echo "camylanet_raw=$camylanet_raw"
echo "camylanet_preprocessed=$camylanet_preprocessed"
echo "camylanet_results=$camylanet_results"
