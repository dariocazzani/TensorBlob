# TensorBlob

**An implementation of a Computational Graph in C++ for Machine Learning problems**

**Just clone and run, no installations required.**


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

### Build your own Neural Net for MNIST

* **Load data into train and validation Matrices**
    ```C++
    ...
    #include "mnist/mnistUtils.h"
    ...
    //INPUTS and LABELS
    Eigen::MatrixXd trainData(TRAIN_SIZE, IMG_SIZE*IMG_SIZE);
    Eigen::MatrixXd validData(VALID_SIZE, IMG_SIZE*IMG_SIZE);
    Eigen::MatrixXd trainLabels(TRAIN_SIZE, NUM_CLASSES);
    Eigen::MatrixXd validLabels(VALID_SIZE, NUM_CLASSES);

    getMnistData(trainData, validData, trainLabels, validLabels);
    ```

* **Initialize Trainable Variables - Number varies on the number of hidden layers**
    ```C++
    Eigen::MatrixXd weights1 = Eigen::MatrixXd::Random(IMG_SIZE*IMG_SIZE, NUM_HIDDEN);
    Eigen::MatrixXd bias1 = Eigen::MatrixXd::Random(1, NUM_HIDDEN);
    Eigen::MatrixXd weights2 = Eigen::MatrixXd::Random(NUM_HIDDEN, NUM_CLASSES);
    Eigen::MatrixXd bias2 = Eigen::MatrixXd::Random(1, NUM_CLASSES);
    ```
* **Define Graph Nodes and **

## TODO

* [ ] Make test for SGD
* [ ] Make path to mnist independent from where main is called from 
