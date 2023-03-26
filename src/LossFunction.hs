module LossFunction
  ( getLoss,
    getLoss',
  )
where

import           Numeric.LinearAlgebra as La
import           Types

getLoss loss = case loss of
  MSE          -> mse
  CrossEntropy -> crossEntropy
  CrossEntropySoftMax -> crossEntropySoftMax

getLoss' loss = case loss of
  MSE          -> mse'
  CrossEntropy -> crossEntropy'
  CrossEntropySoftMax -> crossEntropySoftMax'

-- MSE (Mean squared error) sum \sum_k(t_k-y_k)^2
mse :: Matrix Double -> Matrix Double -> Double
mse pred tgt = sumElements $ cmap (^ 2) (pred - tgt)

mse2 :: Num c => [c] -> [c] -> c
mse2 pred tgt = foldl (\acc (y, t) -> acc + squearedError (y, t)) 0 $ zip pred tgt
  where
    squearedError (y, t) = (y - t) ^ 2

mse2' :: Num c => [c] -> [c] -> [c]
mse2' pred tgt = zipWith (-) pred tgt

-- MSE derivation
-- mse' :: (Container c e, Num e, Num (c e)) => c e -> c e -> c e
mse' :: Matrix Double -> Matrix Double -> Matrix Double
mse' pred tgt = pred - tgt

crossEntropy :: Matrix Double -> Matrix Double -> Double
crossEntropy pred tgt = negate $ sumElements $ tgt * cmap log pred

crossEntropy' :: Matrix Double -> Matrix Double -> Matrix Double
crossEntropy' pred tgt = (-tgt) / pred
-- softmax x xs = exp x / sum (map exp xs)

cSoftmax :: Matrix Double -> Matrix Double
cSoftmax c = cmap (/ expSum c) c
  where
    expSum c = sumElements (cmap exp c) -- TODO: how does sumElements work

crossEntropySoftMax :: Matrix Double -> Matrix Double -> Double
crossEntropySoftMax pred tgt = negate $ sumElements (tgt * cmap log (cSoftmax pred))

crossEntropySoftMax' :: Matrix Double -> Matrix Double -> Matrix Double
crossEntropySoftMax' pred tgt = cmap ((/expSum) . exp) pred - tgt
  where expSum = sumElements (cmap exp pred)
