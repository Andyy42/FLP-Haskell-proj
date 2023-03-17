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
    newWAllSame
  )
where

import           Activations           (getActivation, getActivation')
import           LossFunction          (getLoss, getLoss')
import           Numeric.LinearAlgebra as LA
import           Types                 (Activation (..),
                                        BackpropagationStore (..), DeltasMatrix,
                                        Gradients (..), InMatrix,
                                        Layer (Layer, activation, biases, weights),
                                        LearningRate, Loss (..), LossValue,
                                        NeuralNetwork, OutMatrix, TargetMatrix)

import           System.Random

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
linearW' :: DeltasMatrix -> InMatrix -> Matrix Double
linearW' delta prevZ = cmap (/ batchSize) (tr' prevZ LA.<> delta)
  where
    batchSize = fromIntegral $ rows prevZ

----------------------------------------
-- INITIALIZATION (weghts & biases)

-- New weights where all the digits are same
newWAllSame:: (Int, Int) -> Double -> Matrix Double
newWAllSame (nin, nout) num =
  let k = sqrt (1.0 / fromIntegral nin)
      w = (nin><nout) $ repeat num
  in cmap (k *) w

-- New weights
newW :: (Int, Int) -> Int -> Matrix Double
newW (nin, nout) seed =
  let k = sqrt (1.0 / fromIntegral nin)
      w = (nin><nout) $ randoms (mkStdGen seed)
  in cmap (k *) w

-- New biases
newB :: Int -> Matrix Double
newB nout = (1 >< nout) $ repeat 0.01

----------------------------------------
-- MISC UTILITIES FUNCTIONS

activationFun :: Layer -> Matrix Double -> Matrix Double
activationFun layer = getActivation $ activation layer

activationFun' :: Layer -> Matrix Double -> Matrix Double
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

forward :: NeuralNetwork -> InMatrix -> (OutMatrix, [BackpropagationStore])
forward layers z_in = forwardPass layers z_in []

forwardPass :: NeuralNetwork -> InMatrix -> [BackpropagationStore] -> (OutMatrix, [BackpropagationStore])
forwardPass (layer : layers) z_in backpropStores =
  let u = z_in LA.<> weights layer
      f = activationFun layer
      z = f u
      store = BackpropagationStore {currentLayerU = u, prevLayerZ = z_in}
   in forwardPass layers z $ store : backpropStores
forwardPass [] z_in backpropStores = (z_in, backpropStores)

----------------------------------------
-- BACKWARD PASS

backward :: NeuralNetwork -> [BackpropagationStore] -> DeltasMatrix -> (DeltasMatrix, [Gradients])
backward layers backpropStores delta = backwardPass (reverse layers) backpropStores delta []

-- delta w.r.t. current layer
-- delta: row vector
backwardPass :: NeuralNetwork -> [BackpropagationStore] -> DeltasMatrix -> [Gradients] -> (DeltasMatrix, [Gradients])
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

gradientDescent :: NeuralNetwork -> [Gradients] -> LearningRate -> NeuralNetwork
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

-- TODO: Should not return lossValue as its loss value of and old NN before the update (one-step training!!!)
trainOneStep :: NeuralNetwork -> Loss -> InMatrix -> TargetMatrix -> LearningRate -> (LossValue, NeuralNetwork)
trainOneStep neuralNetwork lossFunction input target lr =
  let (output, backpropStore) = forward neuralNetwork input
      lossValue = getLoss lossFunction output target
      lossDelta = getLoss' lossFunction output target -- gradients of single data are in columns
      (_, gradients) = backward neuralNetwork backpropStore lossDelta
      updatedNeuralNetwork = gradientDescent neuralNetwork gradients lr
   in (lossValue, updatedNeuralNetwork)

trainLoop :: Int -> NeuralNetwork -> Loss -> InMatrix -> TargetMatrix -> LearningRate -> (LossValue, NeuralNetwork)
trainLoop epochs neuralNetwork lossFun trainData targetData lr = last $ take epochs $ iterate trainStep (0.0, neuralNetwork)
  where
    trainStep (_, nn) = trainOneStep nn lossFun trainData targetData lr

batchedTrain :: NeuralNetwork -> Loss -> [InMatrix] -> [TargetMatrix] -> LearningRate -> (LossValue, NeuralNetwork)
batchedTrain neuralNetwork lossFun (x : xs) (t : ts) lr =
  let (accLossValue, accNN) = batchedTrain neuralNetwork lossFun xs ts lr -- recursively call batchedTrain' until we hit base case
      (lossValue, newNN) = trainOneStep accNN lossFun x t lr -- trainOneStep with 'accNN'
   in (accLossValue + lossValue, newNN)
batchedTrain neuralNetwork lossFun _ _ lr = (0, neuralNetwork) -- base case (nothing to train)

batchedTrainLoop :: Int -> NeuralNetwork -> Loss -> [InMatrix] -> [TargetMatrix] -> LearningRate -> (LossValue, NeuralNetwork)
batchedTrainLoop epochs neuralNetwork lossFun trainData targetData lr = train
  where
    train = (batchLoss / fromIntegral (length trainData), trainedNN) -- Just averages batchLoss and returns the tuple..
    (batchLoss, trainedNN) = last $ take epochs $ iterate trainStep (0.0, neuralNetwork) -- Iterate trainStep epochs times feeding it it's output as input
    trainStep (lossValue, nn) = batchedTrain nn lossFun trainData targetData lr -- Batched train step, trains for one epoch on whole batch

-- TODO: Update (forward backward) all in one!!!

----------------------------------------
-- EVALUATE LOSS (sample per sample)

evaluateLoss :: NeuralNetwork -> Loss -> InMatrix -> TargetMatrix -> Double
evaluateLoss nn lossFun xxs tts = (/ n) $ evaluateLoss' nn lossFun xxs tts
  where
    n = fromIntegral $ rows xxs

evaluateLoss' :: NeuralNetwork -> Loss -> InMatrix -> TargetMatrix -> Double
evaluateLoss' nn lossFun xxs tts
  | rows xxs == 0 = 0.0
  | otherwise = getLoss lossFun nnOut t + evaluateLoss nn lossFun xs ts
  where
    (x, xs) = splitRowsAt 1 xxs
    (t, ts) = splitRowsAt 1 tts
    (nnOut, _) = forward nn x
