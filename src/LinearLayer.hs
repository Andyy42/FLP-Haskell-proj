--module LinearLayer where
module Main where

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

type NNInput a = Matrix Double

type NNOutput a = Matrix Double

-- | Linear layer weights gradient
linearW' :: (Numeric b, Fractional b) => Matrix b -> Matrix b -> Matrix b
linearW' z dy = cmap (/ m) (tr' z LA.<> dy)
  where
    m = fromIntegral $ rows z

-- | Linear layer inputs gradient
linearX' :: Numeric t => Matrix t -> Matrix t -> Matrix t
linearX' w dy = dy LA.<> tr' w

-- Nice shorthands for loss functions
-- loss :: Loss -> Matrix Double -> Matrix Double -> Double
-- loss = getLoss

-- loss' :: Loss -> Matrix Double -> Matrix Double -> Matrix Double
-- loss' = getLoss'

-- | Bias gradient
bias' :: Matrix Double -> Matrix Double
bias' f' = tr' $ cmap (/ m) r
  where
    -- Sum elements in each row and return a new matrix
    r = matrix (cols f') $ map sumElements (toColumns f')
    m = fromIntegral $ rows f' 

activationFun :: Layer a -> Matrix Double -> Matrix Double
activationFun layer = getActivation $ activation layer

activationFun' :: Layer a -> Matrix Double -> Matrix Double
activationFun' layer = getActivation' $ activation layer

hadamardProduct :: Matrix Double -> Matrix Double -> Matrix Double
hadamardProduct a b = a * b

forward :: NeuralNetwork Double -> NNInput Double -> (NNOutput Double, [BackpropagationStore Double])
forward layers z_in = forwardPass layers z_in []

forwardPass :: NeuralNetwork Double -> NNInput Double -> [BackpropagationStore Double] -> (NNOutput Double, [BackpropagationStore Double])
forwardPass (layer : layers) z_in backpropStores =
  let u = z_in LA.<> weights layer 
      f = activationFun layer
      z = f u
      store = BackpropagationStore {currentLayerU = u, prevLayerZ= z_in}
   in forwardPass layers z $ store : backpropStores
forwardPass [] z_in  backpropStores = (z_in, backpropStores)

backward :: NeuralNetwork Double -> [BackpropagationStore Double] -> Matrix Double -> (Matrix Double, [Gradients Double])
backward layers backpropStores delta = backwardPass (reverse layers) backpropStores delta []

-- delta w.r.t. current layer
-- delta: row vector
backwardPass :: NeuralNetwork Double -> [BackpropagationStore Double] -> Matrix Double -> [Gradients Double] -> (Matrix Double, [Gradients Double])
backwardPass (layer : layers) (store : backpropStores) delta gradients =
  let f' = activationFun' layer $ currentLayerU store -- data in cols (if batched dim is 'batch_size x data')
      delta_times_f' = hadamardProduct delta f'
      dW = tr' (prevLayerZ store) LA.<> delta_times_f' -- TODO: which z is here???
      dB = bias' delta_times_f' -- Column vector, it does average for each col for batched NNs 
      prevDelta = delta_times_f' LA.<> weights layer    -- \delta F W
      currentGrads =
        Gradients -- TODO: grads
          { dbGradient = dB,
            dwGradient = dW
          }
   in backwardPass layers backpropStores delta (currentGrads : gradients)
backwardPass _ _ delta gradients = (delta, gradients)

gradientDescent :: NeuralNetwork Double -> [Gradients Double] -> Double -> NeuralNetwork Double
gradientDescent (layer : layers) (grad : gradients) lr =
  Layer
    { weights = weights layer - (lr `scale` dwGradient grad),
      biases = biases layer - (lr `scale` dbGradient grad),
      activation = activation layer
    }
    : gradientDescent layers gradients lr
gradientDescent _ _ _ = []

trainOneStep :: NeuralNetwork Double -> Loss -> Matrix Double -> Matrix Double -> Double -> (Double, NeuralNetwork Double)
trainOneStep neuralNetwork lossFunction input target lr =
  let (output, backpropStore) = forward neuralNetwork input
      lossValue = getLoss lossFunction output target
      lossDelta = getLoss' lossFunction output target -- gradients of single data are in columns 
      (_, gradients) = backward neuralNetwork backpropStore lossDelta
      updatedNeuralNetwork = gradientDescent neuralNetwork gradients lr
   in (lossValue, updatedNeuralNetwork)

-- New weights
newW (nin, nout) = do
  let k = sqrt (1.0 / fromIntegral nin)
  w <- randn nin nout
  return (cmap (k *) w)

-- New biases
newB ::  Int -> Matrix Double
newB nout = (nout >< 1) $ repeat 0.01


getNN = let
  (nin, nout) = (3,4)
  b = newB nout
  w = (nin><nout) $ repeat 0.2
  input = (20><nin) $ repeat 0.3 :: Matrix Double
  target = (20><nout) $ repeat 0.31 :: Matrix Double
  neuralNetwork = [Layer {weights = w, biases = b, activation = Sigmoid}]
  in (neuralNetwork,input,target, getLoss' MSE)


-- (neuralNetwork,input,target,loss') =  getNN
-- (output, backpropStore) = forward neuralNetwork input
-- lossDelta = getLoss' MSE output target
-- f' = activationFun' (head neuralNetwork) $ currentLayerU (head backpropStore)
-- (xxx, gradients) = backward neuralNetwork backpropStore lossDelta
--       delta_times_f' = hadamardProduct delta (tr' f')
--       dW = delta_times_f' LA.<> tr' (currentLayerZ store)
--       dB = delta_times_f'
--       prevDelta = delta_times_f' LA.<> weights layer -- \delta F W

--  take epochs $
trainLoop :: Int -> NeuralNetwork Double -> Loss -> Matrix Double -> Matrix Double -> Double -> (Double, NeuralNetwork Double)
trainLoop epochs neuralNetwork lossFun trainData targetData lr = last $ take epochs $ iterate trainStep (0.0, neuralNetwork)
  where
    trainStep (_, nn) = trainOneStep nn lossFun trainData targetData lr

main = do
  trainData <- loadMatrix "data/iris/x.dat"
  --let trainData = trainData'
  targetData <- loadMatrix "data/iris/y.dat"
  --let targetData = trainData'
  let (nin, nout) = (3, 4)

  w1_rand <- newW (nin, nout)
  let b1 = newB nout
  let neuralNetwork = [Layer {weights = w1_rand, biases = b1, activation = Sigmoid}]
  let epochs = 2 

  let lossFunction = MSE
  let input = trainData
  let target = targetData 
  putStr $ show neuralNetwork 
  let (pred0, _) = forward neuralNetwork trainData
  putStrLn $ show $ rows pred0
  putStrLn $ show $ cols pred0
  putStrLn $ show $ rows targetData
  putStrLn $ show $ cols targetData 
  putStrLn $ show $ rows trainData
  putStrLn $ show $ cols trainData 

  print $ takeRows 5 (tr' pred0)
  let (output, backpropStore) = forward neuralNetwork input
  let    lossValue = getLoss lossFunction output target
  print lossValue
  let lossDelta = getLoss' lossFunction output target
  print $ (rows lossDelta )
  print $ (cols lossDelta )
  let  (_, gradients) = backward neuralNetwork backpropStore lossDelta
  print $ show gradients 
  let  updatedNeuralNetwork = gradientDescent neuralNetwork gradients 0.01
  putStr $ show updatedNeuralNetwork 




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
  print $ takeRows 5 (tr' pred0)

  -- putStrLn "Targets"
  -- print $ takeRows 5 targetData 
