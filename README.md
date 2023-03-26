# FLP - Neural Networks Project

How to use the program:
```
Usage: neuralNetworksProj --config FILE [OPTIONS...]
  -h         --help               Print help and exit
  -c FILE    --config=FILE        Required path to configuration file
  -v         --verbose            Verbose mode reads config, create neural network and prints results
  -e NUMBER  --experiment=NUMBER  Enable experiment 0, 1, 2, or 3
```
Config argument specifing the config file is the only required argument.

Experiments are:
0. All experiments
1. Non-batched experiments
2. Batched experiments
3. Batched + shuffling experiments

There are several prepared experiments with defined configs described in next section.

## How to run

Firstly, run `cabal install` to install all the dependencies. Then use `Makefile`
to run predefined experiments use `make`, for example:

```
make run_iris_small
```
All the predefined experiments use configuration in `configs` folder with the same names as the make command. To know more see the configuration.

**Other commands are:**

Iris:
* `run_iris_small` (Sigmoid+MSE)
* `run_iris_big` (3 layers: Tanh,Tanh,Sigmoid + MSE)
* `run_iris_tanh`
* `run_iris_id` 
* `run_iris_relu` 

MNIST: (might be a bit slower to finish)
* `run_mnist_small` (Sigmoid+MSE)
* `run_mnist_big` 
* `run_mnist_42000` (runs on whole 42000 images from mnist)

or shorthands to run all iris/mnist or both:
* `run_all_iris`
* `run_all_mnist`
* `run_all`

See Makefile or configuration file for more info about the experiments. Note that the neural network is very sensitive to initialization and sometimes provides not good results after training.

Example of experiment output:
```
============================================================
Iris dataset ~ bigger NN (Batched + shuffled)
------------------------------------------------------------
Epochs: 2000
Loss: 0.562780046724356
Accuracy: 0.96 (144/150)
------------------------------------------------------------
Showing 5 predictions from neural network:
(5><3)
 [ 0.917494, 0.180156, 1.1805e-2
 , 0.916537, 0.181586, 1.1853e-2
 , 0.916574, 0.181553, 1.1852e-2
 , 0.917048, 0.181317, 1.1847e-2
 , 0.917875, 0.179596, 1.1786e-2 ]

Showing 5 targets:
(5><3)
 [ 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0 ]
============================================================
```

Note that each row represents one prediction or target vector. For iris it's 3-values vector, for MNIST it's 10-values vector. Values in matrix are rounded to 6 decimal places.

## Configuration

The configuation format is pseudo-json:
```
Experiment {
  name: String
  epochs: Int 
  seed: Int 
  batchSize: Int 
  learningRate: Double 
  DataPaths {
      target: FilePath 
      input: FilePath
  }
  lossFunction: String 
  architecture: [
    {
      LinearLayer {
        in: Int
        out: Int
        activation: String 
      }
    ...
    }
  ]
}
```

Where the lossFunction is one of "MSE", "CrossEntropy". The activation can be one of "Tanh", "Sigmoid", "Id", "ReLU".

Note that number of output neurons (or input data) from previous layer must match number of input neurons, and number of outputs of last layer must match dimension of target vector/data!

Examples of configurations are in `configs/` folder.

## Files

All source files are in `src` directory, training data for iris and mnist-numbers datasets are in `data` folder and configuration files which specifies experiments with neural networks are in `configs` folder. Cabal project configuration is in `neuralNetworksProj.cabal`. Build directory for compiled program is in `build`.

```
 build 
 configs
├──  iris_big.conf
├──  iris_small.conf
├──  mnist_big.conf
└──  mnist_small.conf
 data
├──  iris
│  ├──  x.dat
│  └──  y.dat
└──  mnist-numbers
   ├──  x-big.dat
   ├──  x-small.dat
   ├──  y-big.dat
   └──  y-small.dat
 src
├──  Activations.hs
├──  Experiments.hs
├──  LossFunction.hs
├──  Main.hs
├──  NeuralNetwork.hs
├──  ParseInput.hs
└──  Types.hs
 docs 
└──  MATH.md
 Makefile
 neuralNetworksProj.cabal
 README.md
```

## Data

### Iris Dataset
Iris containes only 150 samples, it is good for debugging.
* `data/iris` [Iris with one-hot encoded targets](https://www.kaggle.com/datasets/masterdezign/iris-with-onehotencoded-targets?resource=download&select=y.dat)
### MNIST Numbers Dataset
The MNIST dataset targets were adjusted to be one-hot encoded vectors and adjusted to 
a smaller version (5000 samples) and bigger version (original, 42000 samples) datasets.
* `data/mnist-numbers` [MNIST](http://yann.lecun.com/exdb/mnist/) 



## Source files

This sections explains all the source files:
```
 src
├──  Activations.hs
├──  Experiments.hs
├──  LossFunction.hs
├──  Main.hs
├──  NeuralNetwork.hs
├──  ParseConfig.hs
└──  Types.hs
```

Haskell build dependencies:
```
build-depends:
  base ^>=4.15.1.0,
  hmatrix ^>= 0.20.2,
  parsec ^>= 3.1.14.0,
  random ^>= 1.2.1.1,
  array ^>= 0.5.4.0 
```
Most notably the `hmatrix` for fast matrix implementation on CPU with BLAST/LAPACK in Haskell.

### `` Activations.hs

Contains activations functions and their corresponding derivations which works on `Matrix` data type provided by `hmatrix` library.
The module implements these functions and their derivations: 
* Relu
* Sigmoid
* Tanh
* Identity

It exports getters for these functions and their derivations implemented as:

```haskell
getActivation :: Activation -> (Matrix Double -> Matrix Double)
getActivation activation = case activation of
  Relu    -> cRelu
  Sigmoid -> cSigmoid
  Tanh    -> cTanh
  ID      -> id
```

### `` LossFunction.hs

Similiar to `Activations.hs`, this module implements activation functions and their derivations and exports getter for them.
It implements:
* MSE
* CrossEntropy
* CrossEntropySoftMax (for one-hot encoded data)


### `` NeuralNetwork.hs

Implements forward and backward pass for neural network as well as the training "loop", evaluation of loss or accuracy and auxiliary function (for creating NN, activation functions, derivations, hadamardProduct, ...). 

#### Forward pass

It is implemented with `forward` and auxiliary `forwardPass` functions.
The functions performs a forward pass through a neural network with the
given input and returns the output matrix and a list of *backpropagation stores*
for each layer.

The function recursively calls itself for each layer in the network, applying
the layer's weights and activation function to the input and storing the values needed during backpropagation
in the list of *backpropagation stores*. When there are no more layers left, the
function returns the output matrix from the final layer and the list of backpropagation stores.

##### Some Math and Implementation Details 

Fully-connected linear neural network consists of ${M}^{(\ell)}$ neurons where the input vector $\mathbf{z^{(\ell-1)}}$ has ${M}^{(\ell-1)}$ neurons and has total length $L$. The neurons from previous layer are fully-connected with current layer neurons by weighted connections with weight matrix $\mathbf{W^{(\ell)}}$ which has dimension of ${M}^{(\ell-1)}\times{M}^{(\ell)}$ where rows dim ${M}^{(\ell-1)}$ also represents number of connections to single neuron. We also add bias ${b_n^{(\ell)}}$ to each neuron $n$ connection with bias vector $\mathbf{b}^{(\ell)}$. Vector of values $\mathbf{u}^{(\ell)}$ of neurons in current layer can be denoted as:

$$
\mathbf{u^{(\ell)}} = \mathbf{z^{(\ell-1)}}\mathbf{W^{(\ell)}} + \mathbf{b^{(\ell)}}
$$

Let us denote $f$ as our *activation function* then one layer of neural network with nonlinearity would be written as:

$$
\mathbf{z^{(\ell)}} =f(\mathbf{u}^{(\ell)})= f(\mathbf{z}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)})
$$

and in Haskell the implementation of forward pass for sigle layer with `BackpropagationStore` looks like:

```haskell
forwardPass :: NeuralNetwork -> InMatrix -> [BackpropagationStore] -> (OutMatrix, [BackpropagationStore]) 
forwardPass (layer : layers) z_in backpropStores =
  let u = z_in LA.<> weights layer
      f = activationFun layer
      z = f u
      store = BackpropagationStore {currentLayerU = u, prevLayerZ = z_in}
   in forwardPass layers z $ backpropStores ++ [store]
forwardPass [] z_in backpropStores = (z_in, backpropStores)
```

#### Backward pass

The backward pass is implemented with `backward` and auxiliary `backwardPass` function.

The functions perform the backward pass through a single layer of the neural network,
computing the delta matrix and gradients using the given backpropagation store and delta matrix.

The functions applie the backpropagation formula to compute the delta matrix and
gradients for the current layer, and stores the gradients in the list. It then
calls itself recursively with the next layer and delta matrix. When there are no
more layers left, the function returns the final delta matrix and list of gradients.

##### Some Math and Implementation Details 

During _backpropagation_ we want to update every variable by a *reasonable amount* with _gradient descent_
so we minimize the overall value of the total error $E$ obtained from the loss function.
So for each layer's weight matrix $\mathbf{W}^{(\ell)}$ and bias vector $\mathbf{b}^{(\ell)}$ we want to calculate gradient w.r.t error $E$ as:

$$
\frac{\partial{E}}{\partial\mathbf{W}^{(\ell)}} =
\delta^{(\ell)}
\frac{{\partial\mathbf{z}^{(\ell
)}}}{\partial\mathbf{W}^{(\ell)}} =
\bigl(
    \mathbf{z}^{(\ell-1)}
\bigr)^T
\delta^{(\ell)}
\mathbf{F}^{(\ell)}
$$

$$
\frac{\partial{E}}{\partial\mathbf{b}^{(\ell)}} =
\delta^{(\ell)}
\frac{{\partial\mathbf{z}^{(\ell
)}}}{\partial\mathbf{b}^{(\ell)}} =
\delta^{(\ell)}
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}}\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}} =
\delta^{(\ell)}
\mathbf{F}^{(\ell)}
\mathbf{I}
$$

$$
\delta^{(\ell)}=
\delta^{(\ell+1)}
\frac{{\partial\mathbf{z}^{(\ell+1)}}}{\partial\mathbf{z}^{(\ell)}} =
\delta^{(\ell+1)} \mathbf{F}^{(\ell+1)} \mathbf{W^{(\ell+1)}}
$$

$$
\delta^{(L)}=
\frac{\partial{E}}{\partial\mathbf{z}^{(L)}}
$$

**Caveats:** Implementation detail for: $\delta^{(\ell+1)}\mathbf{F}^{(\ell+1)}$ can be computed as the _Hadamard product_ (pair-wise multiplication)

It looks indimidating but all we need is to rewrite it to Haskell to make it work.
It is done as:

```haskell
backwardPass :: NeuralNetwork -> [BackpropagationStore] -> DeltasMatrix -> [Gradients] -> (DeltasMatrix, [Gradients])
backwardPass (layer : layers) (store : backpropStores) delta gradients =
  let f' = activationFun' layer $ currentLayerU store -- data in cols (if batched dim is 'batch_size x data')
      delta_times_f' = hadamardProduct delta f'
      batchSize = fromIntegral $ rows delta_times_f' :: Double
      dW = linearW' delta_times_f' (prevLayerZ store)
      dB = bias' delta_times_f' -- Column vector, it does average for each col for batched NNs
      prevDelta = delta_times_f' LA.<> tr' (weights layer) -- \delta F W^T
      currentGrads = Gradients {
        dbGradient = dB, dwGradient = dW
      }
   in backwardPass layers backpropStores prevDelta (currentGrads : gradients)
backwardPass _ _ delta gradients = (delta, gradients)

-- For the batched input the gradients db_i are obtained as column-wise mean where columns
-- represent batches.
bias' :: Matrix Double -> Matrix Double
bias' f' = tr' $ cmap (/ batchSize) dB
  where
    -- Sum elements in each row and return a new matrix
    dB = matrix (cols f') $ map sumElements (toColumns f')
    batchSize = fromIntegral $ rows f'

-- | Weights gradient. Calculates derivation of linear layer output w.r.t weight matrix W.
-- Returns matrix with gradients dw_ij.
--
-- For the batched input the gradients dw_ij in gradient matrix are divided by the number of batches.
linearW' :: DeltasMatrix -> InMatrix -> Matrix Double
linearW' delta prevZ = cmap (/ batchSize) (tr' prevZ LA.<> delta)
  where
    batchSize = fromIntegral $ rows prevZ
```

#### Stochastic Gradient Descent (SGD) 

Apply stochastic gradient descent (SGD) to update the weights and biases of
the neural network.

Gradient descent uses simple update rule to update neural network parameters $\theta=\{\mathbf{W}^{(\ell)}, \mathbf{b}^{(\ell)}\}$ as

$$
\theta :=\theta-\gamma\nabla_\theta{J(\theta)}
$$

Given a neural network, a list of gradients (one for each layer), and a learning rate,
the function computes new weights and biases by subtracting the product of the learning
rate and the corresponding gradient from the current weights and biases.

```haskell
gradientDescent ::
  NeuralNetwork -> [Gradients] -> LearningRate -> NeuralNetwork
gradientDescent (layer : layers) (grad : gradients) lr =
  Layer
    { weights = weights layer - (lr `scale` dwGradient grad),
      biases = biases layer - (lr `scale` dbGradient grad),
      activation = activation layer
    }
    : gradientDescent layers gradients lr
gradientDescent _ _ _ = []
```
where `scale` is used to multiply matrix by a scalar.

#### Training

The core idea of a single traning step is to:
1. perform forward pass and obtain the `output` and values used in backpropagation (`backpropStore`)
2. calculate gradients of loss function (`lossDelta`) with outputs of NN and target data
3. use the greadients from step 2. and values for backpropagation from step 1. to
compute all the gradients with backward pass 
4. use the gradients from step 3. top update the whole neural network by SGD

This algorithm is implemented in `trainOneStep` function. For training we just need to `iterate` this function for all data "epoch times". This is done in `trainLoop` or `batchedTrainLoop` function. 

Note that there are several options for training:
* **Non-Batched training** - whole matrix of training data and target data is used for forward, backward pass and subsequent update. Not usable for bigger data sets which won't fit into memory.
* **Batched training** - data are split into mini-batches such as: `([InMatrix], [TargetMatrix])` and training is done on these mini-batches
* **Batched training with shuffle** - same as the above except data are shuffled for each epoch to achieve more optimal results (so the NN won't try to learn the order of training samples). Shuffling is done by function `shuffle'` taken from https://wiki.haskell.org/Random_shuffle (it is not as fast as other methods though).

#### Evaluating

After the neural network is trained for some number of epochs it is important to evaluate it. This is done by `evaluateLoss` and `evaluateAccuracy` functions. Loss uses loss of the neural network which was used for training, accuracy simple returns percentage (and count) of values where maximum in NN output matched the max of target (target is one-hot-encoded vector so vector with single 1 and 0 elsewhere)

Note that evaluation should be done on validation and test data split. But here, only the training data are used (which is OK here but not for some more serious machine learning)

### `` Types.hs

Implements various types used by or shared among other modules.

### `` ParseConfig.hs

Parses config and returns `Experiment` record data sctructure.

### `` Experiments.hs

Defines experiments with reporting to stdout. Implements experiments for:
* All experiments - `doAllExperiments`
* Not batched experiment - `doExperimentNotBatched`
* Batched experiment - `doExperimentBatched`
* Batched + shuffling experiment - `doExperimentBatchedShuffled`


### `` Main.hs

Parses the command line arguments and calls other functions for training neural networks with `doExperiments`.


