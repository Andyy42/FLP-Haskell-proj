module Main where

import           Control.Monad         (when)
import           Experiments           (doAllExperiments, doExperimentBatched,
                                        doExperimentBatchedShuffled,
                                        doExperimentNotBatched, testPrintConfig)
import           ParseConfig           (parseExperiment)
import           System.Console.GetOpt
import           System.Environment    (getArgs)
import           System.Exit           (exitFailure)
import           Types                 (Experiment (..))

-- TODO: Process args

data Flag = Help | Config FilePath | Verbose | Experiments String deriving (Show, Eq)

options :: [OptDescr Flag]
options =
  [ Option ['h'] ["help"] (NoArg Help) "Print help and exit"
  , Option ['c'] ["config"] (ReqArg Config "FILE") "Required path to configuration file"
  , Option ['v'] ["verbose"] (NoArg Verbose) "Verbose mode reads config, create neural network and prints results"
  , Option ['e'] ["experiment"] (ReqArg Experiments "NUMBER") "Enable experiment 0, 1, 2, or 3"
  ]

parseArgs :: [String] -> IO ([Flag], [String])
parseArgs args =
  case getOpt Permute options args of
    (flags, nonOpts, []) -> return (flags, nonOpts)
    (_, _, errs) -> ioError (userError (concat errs ++ usageInfo header options))
  where header = "Usage: neuralNetworksProj --config FILE [OPTIONS...]"

doExperiments :: Maybe String -> Experiment -> IO()
doExperiments Nothing exp = putStrLn "There is no experiment to do!"
doExperiments (Just expNum) exp = case expNum of
  "0" -> doAllExperiments exp
  "1" -> doExperimentNotBatched exp
  "2" -> doExperimentBatched exp
  "3" -> doExperimentBatchedShuffled exp
  _   -> putStrLn ("Invalid experiment number: " ++ show expNum) >> exitFailure

readConfig :: FilePath -> IO Experiment
readConfig filename = do
  input <- readFile filename
  case parseExperiment input of
    Left err            -> print err >> exitFailure
    Right newExperiment -> return newExperiment

main :: IO ()
main = do
  args <- getArgs
  (flags, nonOpts) <- parseArgs args

  let configFile = foldl (\acc flag -> case flag of Config fp -> Just fp; _ -> acc) Nothing flags
  if Help `elem` flags || configFile == Nothing
    then putStrLn (usageInfo "Usage: neuralNetworksProj --config FILE [OPTIONS...]" options)
    else do
      let experiments = foldl (\acc flag -> case flag of Experiments fp -> Just fp; _ -> acc) Nothing flags
          isVerbose = Verbose `elem` flags
      putStrLn $ "Configuration file: " ++ show configFile
      putStrLn $ "Verbose: " ++ show isVerbose
      putStrLn $ "Experiment: " ++ show experiments
      putStrLn $ "Non-option arguments: " ++ show nonOpts

      expConfig <- maybe exitFailure readConfig configFile
      when isVerbose $ testPrintConfig expConfig
      doExperiments experiments expConfig
