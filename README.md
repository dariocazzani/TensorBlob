# TensorBlob
An implementation of a Computational Graph in C++ for Machine Learning problems

## TODO

* [x] Make tests for MSE.h
* [x] Make tests for Linear.h
* [ ] Make path to mnist independent from where main is called from 

## Instructions

#### Download MNIST dataset
```bash
mkdir -p data
wget -t 45 http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -q --show-progress
wget -t 45 http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -q --show-progress
echo "Uncompressing..."
gunzip -c train-images-idx3-ubyte.gz > data/train-images-idx3-ubyte
gunzip -c train-labels-idx1-ubyte.gz > data/train-labels-idx1-ubyte
rm train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
```

#### Build, run tests and execute main
```bash
make all && bin/main
```
