module Activations
  ( getActivation,
    getActivation',
  )
where

import Numeric.LinearAlgebra as La
import Types

getActivation Relu = cRelu
getActivation Sigmoid = cSigmoid
getActivation Tanh = cTanh
getActivation ID = id

-- getActivation Softmax = cSoftmax

getActivation' Relu = cRelu'
getActivation' Sigmoid = cSigmoid'
getActivation' Tanh = cTanh'
getActivation' ID = cID' -- TODO: Return identity matrix?

-- getActivation' Softmax = softmax' -- TODO: Not implemented

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

-- sigmoid :: (Container c b, Floating b) => c b -> c b
-- sigmoid = cmap fx
--   where
--     fx x_k = 1.0 / (1 + exp (-x_k))

sigmoid' :: Floating a => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)

cSigmoid' :: Matrix Double -> Matrix Double
cSigmoid' = cmap sigmoid'

-- tanh is defined in Prelude.
-- tanh :: Floating a => a -> a
-- tanh x = (exp x - exp (-x)) / exp x + exp (-x)
cTanh :: Matrix Double -> Matrix Double
cTanh = cmap tanh

tanh' :: Floating a => a -> a
tanh' x = 1 - tanh x ^ 2

cTanh' :: Matrix Double -> Matrix Double
cTanh' = cmap tanh'

-- softmax x xs = exp x / sum (map exp xs)

-- -- cSoftmax :: (Container c b, Foldable c, Floating b) => c b -> c b
-- cSoftmax :: Matrix Double -> Matrix Double
-- cSoftmax c = cmap (/ expSum c) c
--   where
--     expSum c = sumElements (cmap exp c) -- TODO: how does sumElements work

-- TODO: How does d_i S_j = ... work here?? How to index?
-- cSoftmax' :: (Container c b, Foldable c, Floating b) => c b -> c b
-- cSoftmax' =
