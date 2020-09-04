# Training Quantized ResNet
Only working for random data right now, need to build pipelines for ImageNet data and update remainder of `resnet.py` to match decorator syntax.

## Build Container Image
Container image is based on the 20.08 release of NVIDIA's [NGC](ngc.nvidia.com) PyTorch image with [Brevitas](https://github.com/Xilinx/brevitas) installed on top of it.
```
docker build -t $USER/qres:20.08 --build-arg tag=20.08 .
```

## Run script
Right now I'm voluming mapping the code into the container because it's under dev, but ideally in production it would be copied in or even pip installed as its own lib
```
docker run --rm -it -v $PWD:/workspace --gpus 1 $USER/qres:20.08 python main.py
```
