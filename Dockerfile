ARG tag=20.08
FROM nvcr.io/nvidia/pytorch:${tag}-py3
RUN pip install git+https://github.com/Xilinx/brevitas.git