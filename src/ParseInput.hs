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

spacedEndline :: Parser ()
spacedEndline = spaces <* endline <* spaces

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
  target <- string "target:" *> spaces *> quotedString <* spacedEndline
  input <- string "input:" *> spaces *> quotedString <* spacedEndline
  string "}"
  return $ DataPaths target input

-- | Parse a Layer block
layerParser :: Parser LinearLayerConfig
layerParser = do
  string "{"
  spaces
  string "LinearLayer {"
  spaces
  inSize <- string "in:" *> spaces *> int <* spacedEndline
  outSize <- string "out:" *> spaces *> int <* spacedEndline
  activation <- string "activation:" *> spaces *> quotedString <* spacedEndline
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
  name <- string "name:" *> spaces *> quotedString <* spacedEndline
  epochs <- string "epochs:" *> spaces *> int <* spacedEndline
  seed <- string "seed:" *> spaces *> int <* spacedEndline
  batchSize <- string "batchSize:" *> spaces *> int <* spacedEndline
  learningRate <- string "learningRate:" *> spaces *> double <* spacedEndline
  dataPaths <- dataPathsParser <* spacedEndline
  lossFunction <- string "lossFunction:" *> spaces *> quotedString <* spacedEndline
  architecture <- string "architecture:" *> spaces *> between (char '[' <* whitespace) (whitespace *> char ']') (many layerParser) <* spacedEndline
  spaces
  string "}"
  return $ Experiment name epochs seed batchSize learningRate dataPaths (fromStrLoss lossFunction) architecture

-- | Parse a configuration file and return an Experiment record
parseExperiment :: String -> Either ParseError Experiment
parseExperiment = parse experimentParser ""
