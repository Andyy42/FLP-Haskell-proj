module NeuralNetwork
  ( forward,
    backward,
    gradientDescent,
    trainOneStep,
    trainLoop,
    newB,
    newW,
  )
where

import Activations (getActivation, getActivation')
import LossFunction (getLoss, getLoss')
import Numeric.LinearAlgebra as LA
import Types
  ( Activation (..),
    BackpropagationStore (..),
    DeltasMatrix,
    Gradients (..),
    InMatrix,
    Layer (Layer, activation, biases, weights),
    LearningRate,
    Loss (..),
    LossValue,
    NeuralNetwork,
    OutMatrix,
    TargetMatrix,
  )

-- Approach 1
-- do forward and collect values for each layer
-- do backward with collcted stuff on current NN, return new updated NN

-- Approach 1.1
-- Use some temporal stores? Store monad??

-- Approach 2
-- mix forward and backward together:
-- 0. Define update of single layer as:
-- 1. forward
-- 2. ask for dy by updating next layer (recursion to 0.)
-- 3. calculate dw db dy, return updated NN and dy

-- | Bias gradient. Calculates derivation of linear layer output w.r.t bias vector b.
--
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
linearW' :: DeltasMatrix Double -> InMatrix Double -> Matrix Double
linearW' delta prevZ = cmap (/ batchSize) (tr' delta LA.<> prevZ)
  where
    batchSize = fromIntegral $ rows prevZ

activationFun :: Layer a -> Matrix Double -> Matrix Double
activationFun layer = getActivation $ activation layer

activationFun' :: Layer a -> Matrix Double -> Matrix Double
activationFun' layer = getActivation' $ activation layer

hadamardProduct :: Matrix Double -> Matrix Double -> Matrix Double
hadamardProduct a b = a * b

-- FORWARD PASS

forward :: NeuralNetwork Double -> InMatrix Double -> (OutMatrix Double, [BackpropagationStore Double])
forward layers z_in = forwardPass layers z_in []

forwardPass :: NeuralNetwork Double -> InMatrix Double -> [BackpropagationStore Double] -> (OutMatrix Double, [BackpropagationStore Double])
forwardPass (layer : layers) z_in backpropStores =
  let u = z_in LA.<> weights layer
      f = activationFun layer
      z = f u
      store = BackpropagationStore {currentLayerU = u, prevLayerZ = z_in}
   in forwardPass layers z $ store : backpropStores
forwardPass [] z_in backpropStores = (z_in, backpropStores)

-- FORWARD PASS

backward :: NeuralNetwork Double -> [BackpropagationStore Double] -> DeltasMatrix Double -> (DeltasMatrix Double, [Gradients Double])
backward layers backpropStores delta = backwardPass (reverse layers) backpropStores delta []

-- delta w.r.t. current layer
-- delta: row vector
backwardPass :: NeuralNetwork Double -> [BackpropagationStore Double] -> DeltasMatrix Double -> [Gradients Double] -> (DeltasMatrix Double, [Gradients Double])
backwardPass (layer : layers) (store : backpropStores) delta gradients =
  let f' = activationFun' layer $ currentLayerU store -- data in cols (if batched dim is 'batch_size x data')
      delta_times_f' = hadamardProduct delta f'
      batchSize = fromIntegral $ rows delta_times_f' :: Double
      dW = linearW' delta_times_f' (prevLayerZ store)
      dB = bias' delta_times_f' -- Column vector, it does average for each col for batched NNs
      prevDelta = delta_times_f' LA.<> weights layer -- \delta F W
      currentGrads =
        Gradients -- TODO: grads
          { dbGradient = dB,
            dwGradient = dW
          }
   in backwardPass layers backpropStores delta (currentGrads : gradients)
backwardPass [] _ delta gradients = (delta, gradients)
backwardPass _ [] delta gradients = (delta, gradients)

-- STOCHASTIC GRADIENT DESCENT (SGD)

gradientDescent :: NeuralNetwork Double -> [Gradients Double] -> LearningRate -> NeuralNetwork Double
gradientDescent (layer : layers) (grad : gradients) lr =
  Layer
    { weights = weights layer - (lr `scale` dwGradient grad),
      biases = biases layer - (lr `scale` dbGradient grad),
      activation = activation layer
    }
    : gradientDescent layers gradients lr
gradientDescent [] _ _ = []
gradientDescent _ [] _ = []

trainOneStep :: NeuralNetwork Double -> Loss -> InMatrix Double -> TargetMatrix Double -> LearningRate -> (LossValue, NeuralNetwork Double)
trainOneStep neuralNetwork lossFunction input target lr =
  let (output, backpropStore) = forward neuralNetwork input
      lossValue = getLoss lossFunction output target
      lossDelta = getLoss' lossFunction output target -- gradients of single data are in columns
      (_, gradients) = backward neuralNetwork backpropStore lossDelta
      updatedNeuralNetwork = gradientDescent neuralNetwork gradients lr
   in (lossValue, updatedNeuralNetwork)

--  take epochs $
trainLoop :: Int -> NeuralNetwork Double -> Loss -> InMatrix Double -> TargetMatrix Double -> LearningRate -> (LossValue, NeuralNetwork Double)
trainLoop epochs neuralNetwork lossFun trainData targetData lr = last $ take epochs $ iterate trainStep (0.0, neuralNetwork)
  where
    trainStep (_, nn) = trainOneStep nn lossFun trainData targetData lr

-- New weights
newW :: (Int, Int) -> IO (Matrix Double)
newW (nin, nout) = do
  let k = sqrt (1.0 / fromIntegral nin)
  w <- randn nin nout
  return (cmap (k *) w)

-- New biases
newB :: Int -> Matrix Double
newB nout = (1 >< nout) $ repeat 0.01
