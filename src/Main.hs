module Main where

import Activations (getActivation, getActivation')
import LossFunction (getLoss, getLoss')
import NeuralNetwork
  ( backward,
    forward,
    gradientDescent,
    newB,
    newW,
    trainLoop,
    trainOneStep,
  )
import Numeric.LinearAlgebra as LA
import Types
  ( Activation (..),
    BackpropagationStore,
    DeltasMatrix,
    Gradients,
    InMatrix,
    Layer (Layer, activation, biases, weights),
    LearningRate,
    Loss (..),
    LossValue,
    NeuralNetwork,
    OutMatrix,
    TargetMatrix,
  )

main = do
  trainData <- loadMatrix "data/iris/x.dat"
  -- let trainData = trainData'
  targetData <- loadMatrix "data/iris/y.dat"
  -- let targetData = trainData'
  let (nin, nout) = (4, 3)

  w1_rand <- newW (nin, nout)
  let b1 = newB nout
  let neuralNetwork = [Layer {weights = w1_rand, biases = b1, activation = Sigmoid}]
  let epochs = 1000

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
  print $ takeRows 5 (pred1)
  putStrLn "Matching original target data:"
  print $ takeRows 5 (targetData)

-- putStrLn "Targets"
-- print $ takeRows 5 targetData

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
