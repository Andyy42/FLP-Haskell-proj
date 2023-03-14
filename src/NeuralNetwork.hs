module NeuralNetwork
  ( forward,
    backward,
    gradientDescent,
    trainOneStep,
    trainLoop,
    batchedTrainLoop,
    createBatches,
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

----------------------------------------
-- DERIVATIONS OF LINEAR LAYER

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
linearW' delta prevZ = cmap (/ batchSize) (tr' prevZ LA.<> delta)
  where
    batchSize = fromIntegral $ rows prevZ

----------------------------------------
-- INITIALIZATION (weghts & biases)

-- New weights
newW :: (Int, Int) -> IO (Matrix Double)
newW (nin, nout) = do
  let k = sqrt (1.0 / fromIntegral nin)
  w <- randn nin nout
  return (cmap (k *) w)

-- New biases
newB :: Int -> Matrix Double
newB nout = (1 >< nout) $ repeat 0.01

----------------------------------------
-- MISC UTILITIES FUNCTIONS

activationFun :: Layer a -> Matrix Double -> Matrix Double
activationFun layer = getActivation $ activation layer

activationFun' :: Layer a -> Matrix Double -> Matrix Double
activationFun' layer = getActivation' $ activation layer

hadamardProduct :: Matrix Double -> Matrix Double -> Matrix Double
hadamardProduct a b = a * b

-- `takeRows` and `dropRows` is Already in LA
-- takeRows :: Element t => Int -> Matrix t -> Matrix t
-- takeRows n m = m ?? (Take n, All)
-- dropRows :: Element t => Int -> Matrix t -> Matrix t
-- dropRows n m = m ?? (Drop n, All)

splitRowsAt :: Element t => Int -> Matrix t -> (Matrix t, Matrix t)
splitRowsAt n m = (takeRows n m, dropRows n m)

createBatches :: Int -> Matrix Double -> [Matrix Double]
createBatches n m = if rows m /= 0 then takeRows n m : createBatches n (dropRows n m) else []

--  train, validation, test split
createDataSplit :: (Double, Double, Double) -> Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
createDataSplit (trainR, validR, testR) m = (trainSplit, validSplit, testSplit)
  where
    (trainSplit, rest) = makeSplit trainR m
    (validSplit, testSplit) = splitRowsAt validN rest
    validN = round $ validR * fromIntegral (rows m)

makeSplit :: Double -> Matrix Double -> (Matrix Double, Matrix Double)
makeSplit ratio m = splitRowsAt n m
  where
    n = round $ ratio * fromIntegral (rows m)

----------------------------------------
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

----------------------------------------
-- BACKWARD PASS

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

----------------------------------------
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

----------------------------------------
-- TRAINING FUNCTIONS

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

batchedTrainLoop :: Int -> NeuralNetwork Double -> Loss -> [InMatrix Double] -> [TargetMatrix Double] -> LearningRate -> (LossValue, NeuralNetwork Double)
batchedTrainLoop epochs neuralNetwork lossFun trainData targetData lr = train
  where
    train = (batchLoss / fromIntegral (length trainData), trainedNN) -- Just averages batchLoss and returns the tuple..
    (batchLoss, trainedNN) = last $ take epochs $ iterate trainStep (0.0, neuralNetwork) -- Iterate trainStep epochs times feeding it it's output as input
    trainStep (lossValue, nn) = batchedTrain nn lossFun trainData targetData lr -- Batched train step, trains for one epoch on whole batch

batchedTrain :: NeuralNetwork Double -> Loss -> [InMatrix Double] -> [TargetMatrix Double] -> LearningRate -> (LossValue, NeuralNetwork Double)
batchedTrain neuralNetwork lossFun (x : xs) (t : ts) lr =
  let (accLossValue, accNN) = batchedTrain neuralNetwork lossFun xs ts lr -- recursively call batchedTrain' until we hit base case
      (lossValue, newNN) = trainOneStep accNN lossFun x t lr -- trainOneStep with 'accNN'
   in (accLossValue + lossValue, newNN)
batchedTrain neuralNetwork lossFun _ _ lr = (0, neuralNetwork) -- base case (nothing to train)

-- TODO: Batched train loop

----------------------------------------
-- EVALUATE LOSS (sample per sample)

evaluateLoss :: NeuralNetwork Double -> Loss -> InMatrix Double -> TargetMatrix Double -> Double
evaluateLoss nn lossFun xxs tts = (/n) $ evaluateLoss' nn lossFun xxs tts
  where n = fromIntegral $ rows xxs

evaluateLoss' :: NeuralNetwork Double -> Loss -> InMatrix Double -> TargetMatrix Double -> Double
evaluateLoss' nn lossFun xxs tts 
  | rows xxs == 0 = 0.0
  | otherwise = getLoss lossFun nnOut t + evaluateLoss nn lossFun xs ts
  where 
    (x,xs) = splitRowsAt 1 xxs
    (t,ts) = splitRowsAt 1 tts 
    (nnOut, _ ) = forward nn x 
