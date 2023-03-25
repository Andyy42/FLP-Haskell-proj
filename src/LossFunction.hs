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

getLoss' loss = case loss of
    MSE          ->  mse'
    CrossEntropy -> crossEntropy'

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

-- crossEntropy :: (Container c e, Num e, Num (c e)) => c e -> c e -> e
-- Cross Entropy -\sum_k(t_k* log(y_k))
-- crossEntropy :: (Container c a, Num (c a), Floating a) => c a -> c a -> a
crossEntropy :: Matrix Double -> Matrix Double -> Double
crossEntropy pred tgt = negate $ sumElements $ tgt * cmap log pred

-- crossEntropy' :: (Container c a, Num (c a), Floating a) => c a -> c a -> c a
-- Cross Entropy derivative for each k: (-\frac{t_k}{y_k})_k
-- crossEntropy' :: (Container c b, Num b, Num (c b)) => c b -> c b -> c b
crossEntropy' :: Matrix Double -> Matrix Double -> Matrix Double
crossEntropy' pred tgt = -tgt * cmap (^ (-1)) pred
