#!/bin/bash
# export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'

# run this script in the root directory of the project (inside the docker container)

CUDA_VERSION_LIST=("12.2" "11.8")
PYTHON_VERSION_LIST=(cp38 cp39 cp310 cp311 cp312)

for CUDA_VERSION in "${CUDA_VERSION_LIST[@]}"
do
    # remove `.` from CUDA_VERSION
    CUDA_VERSION_TAG=cu$(echo $CUDA_VERSION | tr -d .)
    
    # reset symbolic link to the current CUDA version
    rm /usr/local/cuda
    ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda

    for PYTHON_VERSION in "${PYTHON_VERSION_LIST[@]}"
    do
        echo "Building wheel for CUDA ${CUDA_VERSION} and Python ${PYTHON_VERSION}"
        /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python setup.py bdist_wheel

        # If success, get the latest wheel file in dist/
        if [ $? -eq 0 ]; then
            WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
            WHEEL_FILE_NEW=$(echo $WHEEL_FILE | sed -E "s/(.*[a-z]-[0-9\.]+)-/\1+$CUDA_VERSION_TAG-/")
            mv $WHEEL_FILE $WHEEL_FILE_NEW
        else
            echo "Failed to build wheel for CUDA ${CUDA_VERSION} and Python ${PYTHON_VERSION}"
            exit 1
        fi
    done
done