# author
MAINTAINER Vivian Cheng
# info
LABEL version="1.0"
LABEL description="ml_phaas"

FROM nvcr.io/nvidia/pytorch:20.06-py3
ENV PYTHON_VERSION=3.8

## update pip
RUN pip install --upgrade pip
WORKDIR /ml_phaas

COPY environment.yml ./
RUN conda env create -f environment.yml

COPY *.py ./
COPY images ./images/
COPY data ./data/
COPY point2mesh ./point2mesh/
COPY pymeshfix ./pymeshfix/
COPY Manifold ./Manifold/
COPY pypoisson ./pypoisson/
COPY .gitignore ./
COPY README.md ./

