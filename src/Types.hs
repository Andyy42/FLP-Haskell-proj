module Types
  ( NeuralNetwork (..),
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
    DataPaths (..),
    LinearLayerConfig (..),
    Experiment (..),
  )
where

import Numeric.LinearAlgebra (Matrix, cols, rows)

type InMatrix = Matrix Double

type OutMatrix = Matrix Double

type TargetMatrix = Matrix Double

type DeltasMatrix = Matrix Double

type LearningRate = Double

type LossValue = Double

-- TODO: lookup function for activations
data Activation = Relu | Sigmoid | Tanh | ID deriving (Show)

-- \| Softmax

-- TODO: lookup function for loss functions
data Loss = MSE | CrossEntropy deriving (Show)

-- Layer containes:   weights,   biases,   activation (non-linearity)
data Layer = Layer
  { weights :: Matrix Double,
    biases :: Matrix Double,
    activation :: Activation
  }

instance Show Layer where
  show layer =
    "LinearLayer: out="
      ++ show (cols $ weights layer)
      ++ " in="
      ++ show (rows $ weights layer)
      ++ "\n"
      ++ "   Activation:"
      ++ show (activation layer)
      ++ "\n"

-- NeuralNetwork (fully-connected) is made of layers
type NeuralNetwork = [Layer]

-- instance Show (NeuralNetwork a) where
--   show = showLayer
--     where
--         showLayer [x:xs] = show x ++ showLayer xs

-- Gradients for biases 'db' and weights 'dw"
data Gradients = Gradients
  { dbGradient :: Matrix Double,
    dwGradient :: Matrix Double 
  }
  deriving (Show)

data BackpropagationStore = BackpropagationStore
  { prevLayerZ :: Matrix Double,
    currentLayerU :: Matrix Double
  }

-- type BackpropagationStoreValues a = [BackpropagationStore a]

data TrainData = TrainData
  { inTrain :: [Matrix Double],
    tgtTrain :: [Matrix Double],
    inValid :: [Matrix Double],
    tgtValid :: [Matrix Double]
  }

-- | The DataPaths record type
data DataPaths = DataPaths
  { dpTarget :: String,
    dpInput :: String
  }
  deriving (Show)

-- | The Layer type
data LinearLayerConfig = LinearLayerConfig
  { llIn :: Int,
    llOut :: Int,
    llActivation :: Activation
  }
  deriving (Show)

-- | The Experiment record type
data Experiment = Experiment
  { expName :: String,
    expEpochs :: Int,
    expBatchSize :: Int,
    expLearningRate :: Double,
    expDataPaths :: DataPaths,
    expLossFunction :: Loss,
    expArchitecture :: [LinearLayerConfig]
  }
  deriving (Show)
