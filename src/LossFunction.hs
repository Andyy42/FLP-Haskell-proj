module LossFunction
  ( getLoss,
    getLoss',
  )
where
import Prelude hiding (pred)
import           Numeric.LinearAlgebra as La
import           Types

getLoss :: Loss -> Matrix Double -> Matrix Double -> Double
getLoss loss = case loss of
  MSE          -> mse
  CrossEntropy -> crossEntropy
  CrossEntropySoftMax -> crossEntropySoftMax

getLoss' :: Loss -> Matrix Double -> Matrix Double -> Matrix Double
getLoss' loss = case loss of
  MSE          -> mse'
  CrossEntropy -> crossEntropy'
  CrossEntropySoftMax -> crossEntropySoftMax'

-- MSE (Mean squared error) sum \sum_k(t_k-y_k)^2
mse :: Matrix Double -> Matrix Double -> Double
mse pred tgt = sumElements $ cmap (** 2.0) (pred - tgt)

-- MSE derivation
-- mse' :: (Container c e, Num e, Num (c e)) => c e -> c e -> c e
mse' :: Matrix Double -> Matrix Double -> Matrix Double
mse' pred tgt = pred - tgt

crossEntropy :: Matrix Double -> Matrix Double -> Double
crossEntropy pred tgt = negate $ sumElements $ tgt * cmap log pred

crossEntropy' :: Matrix Double -> Matrix Double -> Matrix Double
crossEntropy' pred tgt = (-tgt) / pred

cSoftmax :: Matrix Double -> Matrix Double
cSoftmax pred = cmap (/ expSum) pred 
  where
    expSum = sumElements (cmap exp pred)

crossEntropySoftMax :: Matrix Double -> Matrix Double -> Double
crossEntropySoftMax pred tgt = negate $ sumElements (tgt * cmap log (cSoftmax pred))

crossEntropySoftMax' :: Matrix Double -> Matrix Double -> Matrix Double
crossEntropySoftMax' pred tgt = cmap ((/expSum) . exp) pred - tgt
  where expSum = sumElements (cmap exp pred)
