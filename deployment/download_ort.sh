#!/bin/bash

# Define version and files
ORT_VERSION="1.16.3"
OS="linux"
ARCH="x64"
ORT_FILE="onnxruntime-${OS}-${ARCH}-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"

# Download
if [ ! -d "onnxruntime-${OS}-${ARCH}-${ORT_VERSION}" ]; then
    echo "Downloading ONNX Runtime ${ORT_VERSION}..."
    curl -L -O ${ORT_URL}
    
    echo "Extracting..."
    tar -xzvf ${ORT_FILE}
    rm ${ORT_FILE}
    echo "Done."
else
    echo "ONNX Runtime already exists."
fi



