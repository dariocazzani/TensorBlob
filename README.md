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

* **Define Variables and PlaceHolders (a.k.a. Input Nodes)**
    ```C++
    // VARIABLES
    Eigen::MatrixXd weights1(IMG_SIZE*IMG_SIZE, NUM_HIDDEN);
    Eigen::MatrixXd bias1(1, NUM_HIDDEN);
    Eigen::MatrixXd weights2(NUM_HIDDEN, NUM_CLASSES);
    Eigen::MatrixXd bias2(1, NUM_CLASSES);

    // DEFINE NODES
    Variable W1(weights1);
    Variable b1(bias1);
    Variable W2(weights2);
    Variable b2(bias2);
    Input X;
    Input Y;
    ```
    
* **Initialize Trainable Variables with random values**
    ```C++
    vector<Node *> trainables = {&W1, &b1, &W2, &b2};
    initTrainables(trainables);
    ```
    
* **Build Graph**
    ```C++
    Linear hidden1(&X, &W1, &b1);
    Sigmoid outHidden1(&hidden1);
    Linear out(&outHidden1, &W2, &b2);
    SoftXent cost(&out, &Y);

    vector<Node *> graph = {&hidden1, &W1, &b1, &W2, &b2, &X, &outHidden1, &out, &Y, &cost};
    buildGraph(graph);
    ```
    
* **Example of forward pass, backward pass and SGD update**
    ```C++
    //Define a std::map object to feed values to the Network Inputs
    map<Node*, Eigen::MatrixXd> inputMap;

    getBatch(trainData, trainLabels, trainDataBatch, trainLabelBatch);
    inputMap[&X] = trainDataBatch;
    inputMap[&Y] = trainLabelBatch;
    feedValues(inputMap);
    forwardBackward(graph);
    SGD(trainables, LEARNING_RATE);
    trainCost = cost.getValues();
    trainAccuracy = getAccuracy(argMax(trainLabelBatch), argMax(cost.getProbabilities()));
    ```
    
## TODO
* [ ] Make test for SGD
* [ ] Make path to mnist independent from where main is called from 
