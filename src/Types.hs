module Types
  ( NeuralNetwork,
    Layer (..),
    Activation (..),
    Loss (..),
    Gradients (..),
    BackpropagationStore (..),
    InMatrix,
    OutMatrix,
    TargetMatrix,
    DeltasMatrix,
    LearningRate,
    LossValue,
    BatchedData,
    NotBatchedData,
    DataPaths (..),
    LinearLayerConfig (..),
    Experiment (..),
    Datas (..),
  )
where

import           Numeric.LinearAlgebra (Matrix, cols, rows)

type InMatrix = Matrix Double

type OutMatrix = Matrix Double

type TargetMatrix = Matrix Double

type DeltasMatrix = Matrix Double

type LearningRate = Double

type LossValue = Double

-- TODO: lookup function for activations
data Activation = Relu | Sigmoid | Tanh | ID deriving (Show)

-- \| Softmax

data Loss = MSE | CrossEntropy | CrossEntropySoftMax deriving (Show)

-- Layer containes:   weights,   biases,   activation (non-linearity)
data Layer = Layer
  { weights    :: Matrix Double,
    biases     :: Matrix Double,
    activation :: Activation
  }

instance Show Layer where
  show layer =
    "LinearLayer: in="
      ++ show (rows $ weights layer)
      ++ " out="
      ++ show (cols $ weights layer)
      ++ "\n"
      ++ "   Activation:"
      ++ show (activation layer)
      ++ "\n"

-- | NeuralNetwork (fully-connected) made of list of layers
type NeuralNetwork = [Layer]

-- Gradients for biases 'db' and weights 'dw"
data Gradients = Gradients
  { dbGradient :: Matrix Double,
    dwGradient :: Matrix Double
  }
  deriving (Show)

data BackpropagationStore = BackpropagationStore
  { prevLayerZ    :: Matrix Double,
    currentLayerU :: Matrix Double
  }
  deriving (Show)

type BatchedData = ([InMatrix], [OutMatrix])

type NotBatchedData = (InMatrix, OutMatrix)

data Datas = BatchedData BatchedData | NotBatchedData NotBatchedData

-- | The DataPaths record type
data DataPaths = DataPaths
  { dpTarget :: String,
    dpInput  :: String
  }
  deriving (Show)

-- | The Layer type
data LinearLayerConfig = LinearLayerConfig
  { llIn         :: Int,
    llOut        :: Int,
    llActivation :: Activation
  }

instance Show LinearLayerConfig where
  show conf =
    "\n  Linear Layer:"
    ++ " in: " ++ show (llIn conf) ++ " out: " ++ show (llOut conf) ++ "\n"
    ++ "    Activation: " ++ show (llActivation conf)


-- | The Experiment record type
data Experiment = Experiment
  { expName         :: String,
    expEpochs       :: Int,
    expSeed         :: Int,
    expBatchSize    :: Int,
    expLearningRate :: Double,
    expDataPaths    :: DataPaths,
    expLossFunction :: Loss,
    expArchitecture :: [LinearLayerConfig]
  }

instance Show Experiment where
  show e =
    "========================================\n"
    ++ "Experiment: " ++ show (expName e) ++ "\n"
    ++ "Epochs: " ++ show (expEpochs e) ++ "\n"
    ++ "Seed: " ++ show (expSeed e) ++ "\n"
    ++ "Batch Size: " ++ show (expBatchSize e) ++ "\n"
    ++ "Learning Rate: " ++ show (expLearningRate e) ++ "\n"
    ++ "Loss Function: " ++ show (expLossFunction e) ++ "\n"
    ++ "Data Paths: \n"
    ++ "  Targets: " ++ show (dpTarget $ expDataPaths e) ++ "\n"
    ++ "  Inputs: " ++ show (dpInput $ expDataPaths e) ++ "\n"
    ++ "Architecture: " ++ show (expArchitecture e) ++ "\n"
    ++ "========================================\n"
