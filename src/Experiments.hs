{-# LANGUAGE NamedFieldPuns #-}

module Experiments where

import           LossFunction          (getLoss, getLoss')
import           NeuralNetwork         (backward, batchedTrainLoop,
                                        createBatches, evaluateAccuracy,
                                        evaluateLoss, forward, gradientDescent,
                                        newB, newW, newWAllSame, trainLoop,
                                        trainOneStep)
import           Numeric.LinearAlgebra as LA
import           Types                 (Activation (..), BackpropagationStore,
                                        DataPaths (..), Datas (..),
                                        DeltasMatrix, Experiment (..),
                                        Gradients, InMatrix,
                                        Layer (Layer, activation, biases, weights),
                                        LearningRate, LinearLayerConfig (..),
                                        Loss (..), LossValue, NeuralNetwork,
                                        OutMatrix, TargetMatrix)

loadTargets :: Experiment -> IO (Matrix Double)
loadTargets exp = loadMatrix $ dpTarget $ expDataPaths exp

loadInputs :: Experiment -> IO (Matrix Double)
loadInputs exp = loadMatrix $ dpInput $ expDataPaths exp

-- | Creates a linear layer for a neural network with the given configuration
-- parameters. The layer applies the specified activation function to the output.
--
-- The 'LinearLayerConfig' record contains the following fields:
--
-- * 'llIn': an integer representing the size of the input to the layer.
-- * 'llOut': an integer representing the size of the output of the layer.
-- * 'llActivation': a function representing the activation function to be applied
-- to the output of the layer.
--
-- The 'seed' parameter is an integer used to seed the random number generator for
-- initializing the weights of the layer.
--
-- Returns a 'Layer' record containing the weights, biases, and activation function
-- for the created linear layer.
createLinearLayer ::
  -- | The configuration for the linear layer
  LinearLayerConfig ->
  -- | The seed for the random number generator
  Int ->
  -- | The created linear layer
  Layer
createLinearLayer LinearLayerConfig {llIn, llOut, llActivation} seed = Layer {weights = newW (llIn, llOut) seed, biases = newB llOut, activation = llActivation}

-- | Creates a neural network with the specified configuration parameters. The
-- neural network is represented as a list of linear layers, with each layer
-- applying an activation function to the output of the previous layer.
--
-- The 'LinearLayerConfig' list contains the configuration for each linear layer
-- in the network. The 'Int' parameter is used to seed the random number generator
-- for initializing the weights of each layer.
--
-- Returns a 'NeuralNetwork' that contains the created linear layers.
createNN ::
  -- | The configuration for each linear layer in the network
  [LinearLayerConfig] ->
  -- | The seed for the random number generator
  Int ->
  -- | The created neural network
  NeuralNetwork
createNN (x : xs) seed = createLinearLayer x seed : createNN xs (seed + 1)
createNN [] seed       = []

createLinearLayerTEST :: LinearLayerConfig -> Double -> Layer
createLinearLayerTEST LinearLayerConfig {llIn, llOut, llActivation} num = Layer {weights = newWAllSame (llIn, llOut) num, biases = newB llOut, activation = llActivation}

createNNTEST :: [LinearLayerConfig] -> Double -> NeuralNetwork
createNNTEST configs num = map (`createLinearLayerTEST` num) configs

loadBatched :: DataPaths -> Int -> IO ([Matrix Double], [Matrix Double])
loadBatched paths batchN = do
  inputs <- loadBatch dpInput
  targets <- loadBatch dpTarget
  return (inputs, targets)
  where
    loadBatch getter = createBatches batchN `fmap` loadMatrix (getter paths)

printExperimentBlock :: String -> NeuralNetwork -> Loss -> Datas -> Int -> IO ()
printExperimentBlock name nn loss datas epochs =
  putStrLn "============================================================"
    >> putStrLn name
    >> putStrLn "------------------------------------------------------------"
    >> printExperiment nn loss datas epochs
    >> putStrLn "============================================================"

printExperiment :: NeuralNetwork -> Loss -> Datas -> Int -> IO ()
printExperiment nn loss datas@(NotBatchedData (inData, outData)) epochs =
  putStrLn ("Epochs: " ++ show epochs)
    >> printEvaluation nn loss datas
    >> putStrLn "------------------------------------------------------------"
    >> printPredictions nn (takeRows 5 inData)
    >> putStrLn ""
    >> printTargets (takeRows 5 outData)
printExperiment nn loss datas@(BatchedData (inData, outData)) epochs =
  putStrLn ("Epochs: " ++ show epochs)
    >> printEvaluation nn loss datas
    >> putStrLn "------------------------------------------------------------"
    >> printPredictions nn (takeRows 5 $ stack inData)
    >> putStrLn ""
    >> printTargets (takeRows 5 $ stack outData)
  where
    stack (x : xs) = foldr (===) x xs
    stack _        = (0 >< 0) []

printEvaluation :: NeuralNetwork -> Loss -> Datas -> IO ()
printEvaluation nn lossFun datas =
  putStrLn ("Loss: " ++ show (evaluateLoss nn lossFun datas))
    >> putStrLn ("Accuracy: " ++ show accuracy ++ " (" ++ show correct ++ "/" ++ show total ++ ")")
  where
    (accuracy, correct, total) = evaluateAccuracy nn datas

printPredictions :: NeuralNetwork -> InMatrix -> IO ()
printPredictions nn inputs =
  putStrLn ("Showing " ++ show n ++ " predictions from neural network:") >> print (cmap round6dp pred)
  where
    (pred, _) = forward nn inputs
    n = rows inputs
    round6dp x = fromIntegral (round $ x * 1e6) / 1e6 :: Double

printTargets :: OutMatrix -> IO ()
printTargets inputs =
  putStrLn ("Showing " ++ show n ++ " targets:") >> print inputs
  where
    n = rows inputs

testPrintConfig :: Experiment -> IO ()
testPrintConfig exp =
  putStrLn "Loaded configuration:"
    >> print exp
    >> putStrLn ""
    >> putStrLn "Created neural network:"
    >> print nn
    >> putStrLn ""
  where
    lossFun = expLossFunction exp
    seed = expSeed exp
    -- Init Neural Network
    nn = createNN (expArchitecture exp) seed

doAllExperiments :: Experiment -> IO ()
doAllExperiments exp = do
  let lossFun = expLossFunction exp
      seed = expSeed exp
      arch = expArchitecture exp
      paths = expDataPaths exp
      batchSize = expBatchSize exp
      epochs = expEpochs exp
      lr = expLearningRate exp
      name = expName exp
      nn = createNN arch seed
  loadedDatas <- loadBatched paths batchSize
  let datas = BatchedData loadedDatas
  putStrLn "============================================================"
  putStrLn "Evaluation before training:"
  printEvaluation nn lossFun datas
  putStrLn "============================================================"
  putStrLn "============================================================"
  doExperimentNotBatched exp
  doExperimentBatched exp
  doExperimentBatchedShuffled exp

doExperimentBatchedShuffled :: Experiment -> IO ()
doExperimentBatchedShuffled exp = do
  let lossFun = expLossFunction exp
      seed = expSeed exp
      arch = expArchitecture exp
      paths = expDataPaths exp
      batchSize = expBatchSize exp
      epochs = expEpochs exp
      lr = expLearningRate exp
      name = expName exp
      nn = createNN arch seed

  datas@(inputs, targets) <- loadBatched paths batchSize
  let trainedNN = batchedTrainLoop epochs nn lossFun (inputs, targets) lr seed
  putStrLn ""
  printExperimentBlock (name ++ " (Batched + shuffled)") trainedNN lossFun (BatchedData datas) epochs

doExperimentBatched :: Experiment -> IO ()
doExperimentBatched exp = do
  let lossFun = expLossFunction exp
      seed = expSeed exp
      arch = expArchitecture exp
      paths = expDataPaths exp
      batchSize = expBatchSize exp
      epochs = expEpochs exp
      lr = expLearningRate exp
      name = expName exp
      nn = createNN arch seed

  loadedDatas <- loadBatched paths batchSize
  let datas = BatchedData loadedDatas
  let trainedNN = trainLoop epochs nn lossFun datas lr
  putStrLn ""
  printExperimentBlock (name ++ " (Batched + NOT shuffled)") trainedNN lossFun datas epochs

doExperimentNotBatched :: Experiment -> IO ()
doExperimentNotBatched exp = do
  let lossFun = expLossFunction exp
      seed = expSeed exp
      arch = expArchitecture exp
      paths = expDataPaths exp
      batchSize = expBatchSize exp
      epochs = expEpochs exp
      lr = expLearningRate exp
      name = expName exp
      nn = createNN arch seed

  inputs <- loadMatrix $ dpInput paths
  targets <- loadMatrix $ dpTarget paths
  let datas = NotBatchedData (inputs, targets)
  let trainedNN = trainLoop epochs nn lossFun datas lr
  putStrLn ""
  printExperimentBlock (name ++ " (Not Batched)") trainedNN lossFun datas epochs
