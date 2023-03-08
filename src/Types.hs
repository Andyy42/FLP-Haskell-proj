module Types
  ( NeuralNetwork (..),
    Layer (..),
    Activation (..),
    Loss (..),
    Gradients(..),
    BackpropagationStore (..),
  )
where

import Numeric.LinearAlgebra as La

-- TODO: lookup function for activations
data Activation = Relu | Sigmoid | Tanh deriving (Show)

-- \| Softmax

-- TODO: lookup function for loss functions
data Loss = MSE | CrossEntropy

-- Layer containes:   weights,   biases,   activation (non-linearity)
data Layer a = Layer
  { weights :: Matrix a,
    biases :: Matrix a,
    activation :: Activation
  }

instance Show (Layer a) where
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
type NeuralNetwork a = [Layer a]

-- instance Show (NeuralNetwork a) where
--   show = showLayer
--     where
--         showLayer [x:xs] = show x ++ showLayer xs

-- Gradients for biases 'db' and weights 'dw"
data Gradients a = Gradients
  { dbGradient :: Matrix a,
    dwGradient :: Matrix a
  }

data BackpropagationStore a = BackpropagationStore
  { prevInputX :: Matrix a,
    activationFunResult :: Matrix a
  }

-- type BackpropagationStoreValues a = [BackpropagationStore a]

data Experiment = Experiment
  { lr :: Float,
    trainDataPath :: String,
    testDataPath :: String,
    validDataPath :: String,
    modelPath :: String
    -- definition of NN?
  }
