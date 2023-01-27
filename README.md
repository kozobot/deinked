# deinked

Port of https://github.com/vijishmadhavan/SkinDeep from FastAI 1 to FastAI 2.   

# Running Deinked

## Locally

To run the Jupyter Notebooks I use a an extended version of the FastAI docker containers.
https://github.com/fastai/docker-containers.  The changes I've made are:
* Changed the root from fastai to local 
* Changed the base ubuntu image to an NVIDIA/CUDA one (nvidia/cuda:11.3.1-base-ubuntu20.04) for GPU training
* Added an image (-ext) with my dependencies and endpoint
  * Locally install ffmpeg libsm6 libxext6
  * Pip install nvidia-ml-py3 opencv-python Pillow

This is the command that I use to run the customized container.
```commandline
docker run --rm \
    --gpus all --privileged \
    --name fastai --ipc=host \
    -p 8888:8888 \
    -v `pwd`:/home \
    local/fastai-ext \
    jupyter notebook
```

## Collab

Haven't set this up yet

# Setting everything up

## Prepare the training data

Place your training set in data/rawdata.  The tattoo image and clean image should be named <name>_tattoo.jpeg and <name>_clean.jpeg respectively.

Run the Deink - Process Raw Data notebook.  This will prep the images for training and put them in data/tattoo and data/clean respectively.

## Running training

The Deink - Train Model notebook, unsurprisingly, is for training the model.  There is a cell for setting training variables.

## Predicting images

Use the Deink - Predict Image notebook to load the model and process an image of your choosing.