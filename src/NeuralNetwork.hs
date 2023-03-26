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
    newWAllSame,
    evaluateLoss,
    evaluateAccuracy,
  )
where

import           Activations           (getActivation, getActivation')
import           Control.Monad         (forM, liftM)
import           Control.Monad.ST      (ST, runST)
import           Data.Array.ST         (STArray, newListArray, readArray,
                                        writeArray)
import           Data.List             (elemIndex)
import           Data.STRef            (newSTRef, readSTRef, writeSTRef)
import           LossFunction          (getLoss, getLoss')
import           Numeric.LinearAlgebra as LA
import           System.Random
import           Types                 (Activation (..),
                                        BackpropagationStore (..), Datas (..),
                                        DeltasMatrix, Gradients (..), InMatrix,
                                        Layer (Layer, activation, biases, weights),
                                        LearningRate, Loss (..), LossValue,
                                        NeuralNetwork, OutMatrix, TargetMatrix)

------------------------------------------------------------
------------------------------------------------------------
------------------------------------------------------------
-- Taken from: https://wiki.haskell.org/Random_shuffle
-- https://okmij.org/ftp/Haskell/perfect-shuffle.txt

-- | Randomly shuffle a list without the IO Monad
--   /O(N)/
shuffle' :: [a] -> StdGen -> ([a], StdGen)
shuffle' xs gen =
  runST
    ( do
        g <- newSTRef gen
        let randomRST lohi = do
              (a, s') <- liftM (randomR lohi) (readSTRef g)
              writeSTRef g s'
              return a
        ar <- newArray n xs
        xs' <- forM [1 .. n] $ \i -> do
          j <- randomRST (i, n)
          vi <- readArray ar i
          vj <- readArray ar j
          writeArray ar j vi
          return vj
        gen' <- readSTRef g
        return (xs', gen')
    )
  where
    n = length xs
    newArray :: Int -> [a] -> ST s (STArray s Int a)
    newArray n xs = newListArray (1, n) xs

------------------------------------------------------------
------------------------------------------------------------
------------------------------------------------------------

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
newWAllSame :: (Int, Int) -> Double -> Matrix Double
newWAllSame (nin, nout) num =
  let k = sqrt (1.0 / fromIntegral nin)
      w = (nin >< nout) $ repeat num
   in cmap (k *) w

-- New weights
newW :: (Int, Int) -> Int -> Matrix Double
newW (nin, nout) seed =
  let k = sqrt (1.0 / fromIntegral nin)
      w = (nin >< nout) $ randomRs (-1, 1) (mkStdGen seed)
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
createBatches n m = if rows m /= 0 then takeRows nn m : createBatches n (dropRows n m) else []
  where
    nn = min (rows m) n

-- NOTE: Taking more then 'nn' causes neuralNetworksProj: wrong subMatrix ((0,0),(16,4)) of (6x4)

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

-- | Performs a forward pass through a neural network with the given input and
--   returns the output matrix and a list of backpropagation stores for each layer.
--
--   The function uses `forwardPass` function and initiate the backpropagation stores
--   as an empty list.
forward ::
  NeuralNetwork ->                    -- | The neural network to evaluate.
  InMatrix ->                         -- | The input matrix
  (OutMatrix, [BackpropagationStore]) -- | The output matrix and list of backpropagation stores.
forward layers z_in = forwardPass layers z_in []

-- | Helper function that performs a forward pass through a neural network with the
--   given input and returns the output matrix and a list of backpropagation stores
--   for each layer.
--
--   * The first argument is the neural network, represented as a list of layers.
--   * The second argument is the input matrix.
--   * The third argument is the list of backpropagation stores for each previous layer.
--
--   The function recursively calls itself for each layer in the network, applying
--   the layer's weights and activation function to the input and storing the result
--   in the list of backpropagation stores. When there are no more layers left, the
--   function returns the output matrix and the list of backpropagation stores.
--
--   If the neural network has no layers, the function simply returns the input matrix
--   and the list of backpropagation stores (which is usually empty at this point).
forwardPass ::
  NeuralNetwork ->                    -- | The neural network to evaluate.
  InMatrix ->                         -- | The input matrix.
  [BackpropagationStore] ->           -- | The list of backpropagation stores for each previous layer.
  (OutMatrix, [BackpropagationStore]) -- | The output matrix and list of backpropagation stores.
forwardPass (layer : layers) z_in backpropStores =
  let u = z_in LA.<> weights layer
      f = activationFun layer
      z = f u
      store = BackpropagationStore {currentLayerU = u, prevLayerZ = z_in}
   in forwardPass layers z $ backpropStores ++ [store]
forwardPass [] z_in backpropStores = (z_in, backpropStores)

----------------------------------------
-- BACKWARD PASS

-- | Performs the backward pass through a neural network, computing the delta matrix
--   and gradients for each layer using the given backpropagation stores and delta matrix.
--
--   * The first argument is the neural network, represented as a list of layers.
--   * The second argument is the list of backpropagation stores for each layer.
--   * The third argument is the delta matrix for the final layer.
--
--   If the neural network has no layers, the function simply returns the delta matrix
--   and an empty list of gradients.
backward ::
  NeuralNetwork ->            -- | The current layer of the neural network.
  [BackpropagationStore] ->   -- | The backpropagation stores for the previous layers.
  DeltasMatrix ->             -- | The delta matrix for the next layer.
  (DeltasMatrix, [Gradients]) -- | The final delta matrix and list of gradients.
backward layers backpropStores delta = backwardPass (reverse layers) (reverse backpropStores) delta []

-- | Helper function that performs the backward pass through a single layer of the neural
--   network, computing the delta matrix and gradients using the given backpropagation
--   store and delta matrix.
--
--   * The first argument is the current layer of the neural network.
--   * The second argument is the backpropagation store for the current layer.
--   * The third argument is the delta matrix for the next layer.
--   * The fourth argument is the list of gradients computed so far.
--
--   The function applies the backpropagation formula to compute the delta matrix and
--   gradients for the current layer, and stores the gradients in the list. It then
--   calls itself recursively with the next layer and delta matrix. When there are no
--   more layers left, the function returns the final delta matrix and list of gradients.
--
--   If there are no more backpropagation stores or layers, the function simply returns
--   the given delta matrix and list of gradients.
backwardPass ::
  NeuralNetwork ->            -- | The current layer of the neural network.
  [BackpropagationStore] ->   -- | The backpropagation stores for the previous layers.
  DeltasMatrix ->             -- | The delta matrix for the next layer.
  [Gradients] ->              -- | The list of gradients computed so far.
  (DeltasMatrix, [Gradients]) -- | The final delta matrix and list of gradients.
backwardPass (layer : layers) (store : backpropStores) delta gradients =
  let f' = activationFun' layer $ currentLayerU store -- data in cols (if batched dim is 'batch_size x data')
      delta_times_f' = hadamardProduct delta f'
      batchSize = fromIntegral $ rows delta_times_f' :: Double
      dW = linearW' delta_times_f' (prevLayerZ store)
      dB = bias' delta_times_f' -- Column vector, it does average for each col for batched NNs
      prevDelta = delta_times_f' LA.<> tr' (weights layer) -- \delta F W^T
      currentGrads =
        Gradients -- TODO: grads
          { dbGradient = dB,
            dwGradient = dW
          }
   in backwardPass layers backpropStores prevDelta (currentGrads : gradients)
backwardPass [] _ delta gradients = (delta, gradients)
backwardPass _ [] delta gradients = (delta, gradients)

----------------------------------------
-- STOCHASTIC GRADIENT DESCENT (SGD)

-- | Apply stochastic gradient descent (SGD) to update the weights and biases of
-- the neural network.
--
-- Given a neural network, a list of gradients (one for each layer), and a learning rate,
-- the function computes new weights and biases by subtracting the product of the learning
-- rate and the corresponding gradient from the current weights and biases.
--
-- ==== Examples
--
-- Suppose we have a neural network 'net', a list of gradients 'grads', and a learning
-- rate of 0.01. To update the weights and biases of the network, we can call:
--
-- >>> let newNet = gradientDescent net grads 0.01
--
-- This will return a new neural network with updated weights and biases.
--
-- ==== Notes
--
-- * The length of the list of gradients should be equal to the number of layers in the
--   neural network.
-- * The learning rate should be a positive number.
-- * If the neural network has no layers or the list of gradients is empty, the function
--   returns an empty list.
gradientDescent ::
  NeuralNetwork ->  -- | The neural network to update.
  [Gradients] ->    -- | The list of gradients (one for each layer).
  LearningRate ->   -- | The learning rate.
  NeuralNetwork     -- | The updated neural network.
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

-- | Trains a neural network for a single step using the given loss function, input, target, and learning rate.
-- Returns the updated neural network and the loss value (note that the loss value is BEFORE the SGD update).
trainOneStep :: NeuralNetwork -> Loss -> InMatrix -> TargetMatrix -> LearningRate -> (LossValue, NeuralNetwork)
trainOneStep neuralNetwork lossFunction input target lr =
  let (output, backpropStore) = forward neuralNetwork input
      lossValue = getLoss lossFunction output target
      lossDelta = getLoss' lossFunction output target -- gradients of single data are in columns
      (_, gradients) = backward neuralNetwork backpropStore lossDelta
      updatedNeuralNetwork = gradientDescent neuralNetwork gradients lr
   in (lossValue, updatedNeuralNetwork)

-- | Trains a neural network for the specified number of epochs using the given loss function, input, target, and learning rate.
-- Returns the updated neural network.
trainLoop :: Int -> NeuralNetwork -> Loss -> Datas -> LearningRate -> NeuralNetwork
trainLoop epochs neuralNetwork lossFun (NotBatchedData (input, tgt)) lr = last $ take epochs $ iterate trainStep neuralNetwork
  where
    trainStep nn = snd $ trainOneStep nn lossFun input tgt lr
trainLoop epochs nn lossFun (BatchedData datas@(input, tgt)) lr = trainedNN
  where
    trainedNN = last $ take epochs $ iterate trainStep nn -- Iterate trainStep epochs times feeding it it's output as input
    trainStep nn = batchedTrain nn lossFun datas lr -- Batched train step, trains for one epoch on whole batch

-- | Trains a neural network on a batch of data using the given loss function, input, target, and learning rate.
-- Returns the updated neural network.
batchedTrain :: NeuralNetwork -> Loss -> ([InMatrix], [TargetMatrix]) -> LearningRate -> NeuralNetwork
batchedTrain neuralNetwork lossFun (input, tgt) lr =
  foldr (\(x, t) nn -> snd $ trainOneStep nn lossFun x t lr) neuralNetwork (zip input tgt)

-- | Trains a neural network on a batch of data for the specified number of epochs using the given loss function,
-- input, target, learning rate, and random seed (which is used for shuffling the training data during training).
-- Returns the updated neural network.
batchedTrainLoop :: Int -> NeuralNetwork -> Loss -> ([InMatrix], [TargetMatrix]) -> LearningRate -> Int -> NeuralNetwork
batchedTrainLoop epochs nn lossFun datas@(input, tgt) lr seed = trainedNN -- Just averages batchLoss and returns the tuple..
  where
    trainedNN = last $ take epochs iterateTrainStep' -- Iterate trainStep epochs times
    iterateTrainStep' = iterateTrainStep nn lossFun datas lr (mkStdGen seed)

-- | Helper function for 'batchedTrainLoop' that iteratively trains a neural network on a batch of data for one
-- epoch using the given loss function, input, target, and learning rate.
-- Returns a list with updated neural network at each iteration.
iterateTrainStep :: NeuralNetwork -> Loss -> ([InMatrix], [TargetMatrix]) -> LearningRate -> StdGen -> [NeuralNetwork]
iterateTrainStep nn lossFun datas@(input, tgt) lr g = newNN : nextTrainStep
  where
    newNN = batchedTrain nn lossFun datas lr -- Batched train step, trains for one epoch on whole batch
    (shuffledInput, _) = shuffle' input g -- Randmoly shuffle data with random generator g
    (shuffledTgt, newG) = shuffle' tgt g
    nextTrainStep = iterateTrainStep newNN lossFun (shuffledInput, shuffledTgt) lr newG

-----------------------------------------------
-- EVALUATE LOSS (batched or sample per sample)

-- | Calculates the average loss across a given dataset. If the dataset is batched, it evaluates
--   the loss for each batch and returns the average over all batches. If the dataset is not batched,
--   it evaluates the loss for each sample and returns the average over all samples.
evaluateLoss ::
  NeuralNetwork ->  -- | The neural network to evaluate.
  Loss ->           -- | The loss function to use.
  Datas ->          -- | The dataset to evaluate the loss on.
  Double            -- | The average loss over the dataset.
evaluateLoss nn lossFun (BatchedData (inputs, targets)) = if n > 0 then (/ n) $ foldr eval 0.0 $ zip inputs targets else 0.0
  where
    eval e acc = evaluateLoss nn lossFun (NotBatchedData e) + acc
    n = fromIntegral $ length inputs
evaluateLoss nn lossFun (NotBatchedData datas@(xxs, tts)) = if n > 0 then (/ n) $ evaluateLoss' nn lossFun datas else 0.0
  where
    n = fromIntegral $ rows xxs

-- | Helper function for `evaluateLoss` that calculates the average loss for a single input-target pair.
evaluateLoss' ::
  NeuralNetwork ->          -- | The neural network to evaluate.
  Loss ->                   -- | The loss function to use.
  (InMatrix, OutMatrix) ->  -- | The input and target matrices to evaluate the loss on.
  Double                    -- | The average loss over the input-target pair.
evaluateLoss' nn lossFun (xxs, tts)
  | rows xxs == 0 = 0.0
  | otherwise = getLoss lossFun forwardOut tts + evaluateLoss' nn lossFun (xs, ts)
  where
    (x, xs) = splitRowsAt 1 xxs
    (t, ts) = splitRowsAt 1 tts
    (forwardOut, _) = forward nn xxs

-- | Evaluates and prints loss
evaluateAndPrintLoss :: NeuralNetwork -> Loss -> Datas -> IO ()
evaluateAndPrintLoss nn lossFun datas = putStrLn $ "Total loss after training " ++ show (evaluateLoss nn lossFun datas)

---------------------------------------------
-- ACCURACY

-- | Calculates the accuracy of a neural network on a dataset.
--
-- If the dataset contains batched data, the average accuracy over all batches will be returned.
-- If the dataset contains not-batched data, the accuracy will be calculated for the entire dataset.
--
-- The accuracy is returned as a percentage, along with the number of correct predictions and the total number of predictions.
evaluateAccuracy ::
  NeuralNetwork ->            -- | The neural network to evaluate.
  Datas ->                    -- | The input-target data to evaluate the accuracy on.
  (Double, Integer, Integer)  -- | A tuple containing the accuracy as a percentage, the number of correct predictions, and the total number of predictions.
evaluateAccuracy nn (NotBatchedData (inputData, tgtData)) = accuracy (predData, tgtData)
  where
    (predData, _) = forward nn inputData
evaluateAccuracy nn (BatchedData (inputDatas, tgtDatas)) = accuracy (predData, tgtData)
  where
    predData = stack (map (fst . forward nn) inputDatas)
    tgtData = stack tgtDatas
    stack (x : xs) = foldr (===) x xs
    stack _        = (0 >< 0) [] -- Empty matrix if nothing to stack

-- | Calculates the accuracy of a neural network on a pair of predicted and target matrices.
--
-- The accuracy is returned as a percentage, along with the number of correct predictions and the total number of predictions.
accuracy ::
  (InMatrix, Matrix Double) ->  -- | The predicted and target matrices.
  (Double, Integer, Integer)    -- | A tuple containing the accuracy as a percentage, the number of correct predictions, and the total number of predictions.
accuracy (predData, tgtData) = (accuracy, correctCount, fromIntegral (rows tgtData))
  where
    correctCount = foldr (\(pred, tgt) acc -> if maxIndex pred == maxIndex tgt then acc + 1 else acc) 0 zippedRows
    zippedRows = zip (toLists predData) (toLists tgtData)
    maxIndex xs = elemIndex (maximum xs) xs
    n = fromIntegral (rows tgtData)
    accuracy = if n > 0 then fromInteger correctCount / n else n
