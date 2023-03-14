import System.Environment (getArgs)
import Text.Parsec
import Text.Parsec.String (Parser)
import Control.Monad (void)

-- | The Experiment record type
data Experiment = Experiment
  { expName :: String,
    expEpochs :: Int,
    expBatchSize :: Int,
    expLearningRate :: Double,
    expDataPaths :: DataPaths,
    expLossFunction :: String,
    expArchitecture :: [Layer]
  }
  deriving (Show)

-- | The DataPaths record type
data DataPaths = DataPaths
  { dpTarget :: String,
    dpInput :: String
  }
  deriving (Show)

-- | The Layer type
data Layer = LinearLayer
  { llIn :: Int,
    llOut :: Int,
    llActivation :: String
  }
  deriving (Show)

-- | Parse a string enclosed in double quotes
quotedString :: Parser String
quotedString = char '"' *> manyTill anyChar (try $ char '"')

whitespace :: Parser ()
whitespace = void $ many $ oneOf " \n\t"

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
  target <- string "target:" *> spaces *> quotedString <* char '\n'
  spaces
  input <- string "input:" *> spaces *> quotedString <* char '\n'
  spaces
  string "}"
  return $ DataPaths target input

-- | Parse a Layer block
layerParser :: Parser Layer
layerParser = do
  string "{"
  spaces
  string "LinearLayer {"
  spaces
  inSize <- string "in:" *> spaces *> int <* char '\n'
  spaces
  outSize <- string "out:" *> spaces *> int <* char '\n'
  spaces
  activation <- string "activation:" *> spaces *> quotedString <* char '\n'
  spaces
  string "}"
  spaces
  string "}"
  spaces
  return $ LinearLayer inSize outSize activation

-- | Parse an Experiment block
experimentParser :: Parser Experiment
experimentParser = do
  string "Experiment {"
  spaces
  name <- string "name:" *> spaces *> quotedString <* char '\n'
  spaces
  epochs <- string "epochs:" *> spaces *> int <* char '\n'
  spaces
  batchSize <- string "batchSize:" *> spaces *> int <* char '\n'
  spaces
  learningRate <- string "learningRate:" *> spaces *> double <* char '\n'
  spaces
  dataPaths <- dataPathsParser <* char '\n'
  spaces
  lossFunction <- string "lossFunction:" *> spaces *> quotedString <* char '\n'
  spaces
  architecture <- string "architecture:" *> spaces *> between (char '[' <* whitespace) (whitespace*> char ']') (many layerParser) <* char '\n'
  spaces
  string "}"
  return $ Experiment name epochs batchSize learningRate dataPaths lossFunction architecture

-- | Parse a configuration file and return an Experiment record
parseExperiment :: String -> Either ParseError Experiment
parseExperiment input = parse experimentParser "" input


main :: IO ()
main = do
  args <- getArgs
  case args of
    [filename] -> do
      input <- readFile filename
      case parseExperiment input of
        Left err -> print err
        Right exp -> print exp
    _ -> putStrLn "Usage: myprogram config.txt"