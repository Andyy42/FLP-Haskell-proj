module ParseInput (parseExperiment) where

import           Control.Monad      (void)
import           Text.Parsec
import           Text.Parsec.String (Parser)
import           Types              (Activation (..), DataPaths (..),
                                     Experiment (..), LinearLayerConfig (..),
                                     Loss (..))

-- | Parse a string enclosed in double quotes
quotedString :: Parser String
quotedString = char '"' *> manyTill anyChar (try $ char '"')

whitespace :: Parser ()
whitespace = void $ many $ oneOf " \n\t" 

endline :: Parser ()
endline = void $ many $ oneOf " \t" <* char '\n'

fromStrActivation :: String -> Activation
fromStrActivation val
  | val `elem` ["Relu", "ReLu", "ReLU", "relu"] = Relu
  | val `elem` ["Sigmoid", "sigmoid"] = Sigmoid
  | val `elem` ["Tanh", "tanh"] = Tanh
  | otherwise = ID -- TODO: Throw error

fromStrLoss :: String -> Loss
fromStrLoss val
  | val `elem` ["mse", "MSE"] = MSE
  | val `elem` ["CrossEntropy", "CE", "ce"] = CrossEntropy
  | otherwise = MSE -- TODO: Throw error

-- | Parse an integer
int :: Parser Int
int = read <$> many1 digit

-- | Parse a double
double :: Parser Double
double = do
  n <- many1 digit
  char '.'
  f <- many1 digit
  return (read $ n ++ "." ++ f)

-- | Parse a DataPaths block
dataPathsParser :: Parser DataPaths
dataPathsParser = do
  string "DataPaths {"
  spaces
  target <- string "target:" *> spaces *> quotedString <* endline
  spaces
  input <- string "input:" *> spaces *> quotedString <* endline
  spaces
  string "}"
  return $ DataPaths target input

-- | Parse a Layer block
layerParser :: Parser LinearLayerConfig
layerParser = do
  string "{"
  spaces
  string "LinearLayer {"
  spaces
  inSize <- string "in:" *> spaces *> int <* endline
  spaces
  outSize <- string "out:" *> spaces *> int <* endline
  spaces
  activation <- string "activation:" *> spaces *> quotedString <* endline
  spaces
  string "}"
  spaces
  string "}"
  spaces
  return $ LinearLayerConfig inSize outSize $ fromStrActivation activation

-- | Parse an Experiment block
experimentParser :: Parser Experiment
experimentParser = do
  string "Experiment {"
  spaces
  name <- string "name:" *> spaces *> quotedString <* endline  <* spaces
  epochs <- string "epochs:" *> spaces *> int <* endline   <* spaces
  seed <- string "seed:" *> spaces *> int <* endline <* spaces
  batchSize <- string "batchSize:" *> spaces *> int <* endline  <* spaces
  learningRate <- string "learningRate:" *> spaces *> double <* endline  <* spaces
  dataPaths <- dataPathsParser <* endline <* spaces
  lossFunction <- string "lossFunction:" *> spaces *> quotedString <* endline <* spaces
  architecture <- string "architecture:" *> spaces *> between (char '[' <* whitespace) (whitespace *> char ']') (many layerParser) <* char '\n'
  spaces
  string "}"
  return $ Experiment name epochs seed batchSize learningRate dataPaths (fromStrLoss lossFunction) architecture

-- | Parse a configuration file and return an Experiment record
parseExperiment :: String -> Either ParseError Experiment
parseExperiment = parse experimentParser ""
