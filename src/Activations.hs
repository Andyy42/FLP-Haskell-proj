module Activations
  ( getActivation,
    getActivation',
  )
where

import           Numeric.LinearAlgebra as La
import Types ( Activation(..) )

getActivation :: Activation -> (Matrix Double -> Matrix Double)
getActivation activation = case activation of
  Relu    -> cRelu
  Sigmoid -> cSigmoid
  Tanh    -> cTanh
  ID      -> id

-- getActivation Softmax = cSoftmax
getActivation' :: Activation -> (Matrix Double -> Matrix Double)
getActivation' activation = case activation of
  Relu    -> cRelu'
  Sigmoid -> cSigmoid'
  Tanh    -> cTanh'
  ID      -> cID' -- NOTE: Returns vector with ones. (for matrices it should return identity matrix)

-- NOTE: This will work only for vectors!!
cID' :: Matrix Double -> Matrix Double
cID' x = (rows x >< cols x) $ repeat 1.0

relu :: (Floating a, Ord a) => a -> a
relu = max 0

cRelu :: Matrix Double -> Matrix Double
cRelu = cmap relu

relu' :: (Floating a, Ord a) => a -> a
relu' x
  | x > 0 = 1
  | otherwise = 0

cRelu' :: Matrix Double -> Matrix Double
cRelu' = cmap relu'

sigmoid :: Floating a => a -> a
sigmoid x = 1.0 / (1 + exp (-x))

cSigmoid :: Matrix Double -> Matrix Double
cSigmoid = cmap sigmoid

sigmoid' :: Floating a => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)

cSigmoid' :: Matrix Double -> Matrix Double
cSigmoid' = cmap sigmoid'

-- tanh is defined in Prelude.
-- tanh :: Floating a => a -> a
-- tanh x = (exp x - exp (-x)) / exp x + exp (-x)
cTanh :: Matrix Double -> Matrix Double
cTanh = cmap tanh

tanh' :: Double -> Double
tanh' x = 1 - (tanh x ** 2.0)

cTanh' :: Matrix Double -> Matrix Double
cTanh' = cmap tanh'

