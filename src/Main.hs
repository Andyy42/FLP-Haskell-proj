module Main where

import           Experiments        (doAllExperiments, testPrintConfig)
import           ParseInput         (parseExperiment)
import           System.Environment (getArgs)
import           System.Exit        (exitFailure)
import           Types              (Experiment (..))

-- TODO: Process args

main = do
  args <- getArgs
  case args of
    [filename] -> do
      input <- readFile filename
      case parseExperiment input of
        Left err -> print err >> exitFailure
        Right newExperiment ->
          let exp = newExperiment
           in do
                let lossFun = expLossFunction exp
                let seed = expSeed exp

                testPrintConfig exp
                doAllExperiments exp
    _ -> putStrLn "Usage: myprogram [CONFIG]" >> exitFailure
