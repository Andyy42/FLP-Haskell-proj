module NeuralNetwork (
  forward,
  backward,
  gradientDescent,
  trainOneStep,
  trainLoop
  newB,
  newW
)
where

import Activations (getActivation, getActivation')
import LossFunction (getLoss, getLoss')
import Numeric.LinearAlgebra as LA
import Types
  ( Activation (..),
    BackpropagationStore (..),
    Gradients (..),
    Layer (Layer, activation, biases, weights),
    Loss (..),
    NeuralNetwork,
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

-- | Linear layer inputs gradient
linearX' :: Numeric t => Matrix t -> Matrix t -> Matrix t
linearX' w dy = dy LA.<> tr' w

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
linearW' :: (Numeric b, Fractional b) => Matrix b -> Matrix b -> Matrix b
linearW' delta prevZ = cmap (/ batchSize) (tr' prevZ LA.<> delta)
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
      dW = linearW' (prevLayerZ store) delta_times_f'
      dB = bias' delta_times_f' -- Column vector, it does average for each col for batched NNs
      prevDelta = delta_times_f' LA.<> weights layer -- \delta F W
      currentGrads =
        Gradients -- TODO: grads
          { dbGradient = dB,
            dwGradient = dW
          }
   in backwardPass layers backpropStores delta (currentGrads : gradients)
backwardPass _ _ delta gradients = (delta, gradients)

-- STOCHASTIC GRADIENT DESCENT (SGD)

gradientDescent :: NeuralNetwork Double -> [Gradients Double] -> LearningRate -> NeuralNetwork Double
gradientDescent (layer : layers) (grad : gradients) lr =
  Layer
    { weights = weights layer - (lr `scale` dwGradient grad),
      biases = biases layer - (lr `scale` dbGradient grad),
      activation = activation layer
    }
    : gradientDescent layers gradients lr
gradientDescent _ _ _ = []

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

getNN =
  let (nin, nout) = (3, 4)
      b = newB nout
      w = (nin >< nout) $ repeat 0.2
      input = (20 >< nin) $ repeat 0.3 :: InMatrix Double
      target = (20 >< nout) $ repeat 0.31 :: OutMatrix Double
      neuralNetwork = [Layer {weights = w, biases = b, activation = Sigmoid}]
   in (neuralNetwork, input, target, getLoss' MSE)

-- (neuralNetwork,input,target,loss') =  getNN
-- (output, backpropStore) = forward neuralNetwork input
-- lossDelta = getLoss' MSE output target
-- f' = activationFun' (head neuralNetwork) $ currentLayerU (head backpropStore)
-- (xxx, gradients) = backward neuralNetwork backpropStore lossDelta
--       delta_times_f' = hadamardProduct delta (tr' f')
--       dW = delta_times_f' LA.<> tr' (currentLayerZ store)
--       dB = delta_times_f'
--       prevDelta = delta_times_f' LA.<> weights layer -- \delta F W

main = do
  trainData <- loadMatrix "data/iris/x.dat"
  -- let trainData = trainData'
  targetData <- loadMatrix "data/iris/y.dat"
  -- let targetData = trainData'
  let (nin, nout) = (4, 3)

  w1_rand <- newW (nin, nout)
  let b1 = newB nout
  let neuralNetwork = [Layer {weights = w1_rand, biases = b1, activation = Sigmoid}]
  let epochs = 10000

  let lossFunction = MSE
  let input = trainData
  let target = targetData
  putStr $ show neuralNetwork

  let (pred0, _) = forward neuralNetwork trainData

  putStrLn $ "Initial loss " ++ show (getLoss MSE pred0 $ targetData)
  let (lossValue, trainedNN) = trainLoop epochs neuralNetwork MSE trainData targetData 0.01
  -- let (lossValue, trainedNN) = trainOneStep neuralNetwork MSE trainData targetData 0.01

  let (pred1, _) = forward trainedNN trainData
  putStrLn $ show $ rows pred1

  -- let x = iterate ( (\_ nn -> trainOneStep nn MSE trainData targetData 0.01) 0.0 neuralNetwork)

  -- let w1 = last $ descend (grad (dta, tgt)) epochs 0.01 w1_rand

  --    [_, y_pred0] = forward dta w1_rand
  --    [_, y_pred] = forward dta w1
  putStrLn $ "Initial loss " ++ show (getLoss MSE pred0 targetData)
  putStrLn $ "Loss after training " ++ show (getLoss MSE pred1 targetData)

  -- putStrLn "Some predictions by an untrained network:"
  -- print $ takeRows 5 pred0

  putStrLn "Some predictions by a trained network:"
  print $ takeRows 5 (pred0)

-- putStrLn "Targets"
-- print $ takeRows 5 targetData
