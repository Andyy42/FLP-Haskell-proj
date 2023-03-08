module LinearLayer where

import Activations (getActivation, getActivation')
import LossFunction (getLoss, getLoss')
import Numeric.LinearAlgebra as LA
import Types
  ( Activation,
    BackpropagationStore (..),
    Gradients (..),
    Layer (activation, biases, weights),
    Loss,
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
linearW' x dy = cmap (/ m) (tr' x LA.<> dy)
  where
    m = fromIntegral $ rows x

-- | Linear layer inputs gradient
linearX' w dy = dy LA.<> tr' w 


forward :: NeuralNetwork Double -> NNInput Double -> (NNOutput Double, [BackpropagationStore Double])
forward layers x = forwardPass layers x []

forwardPass :: NeuralNetwork Double -> NNInput Double -> [BackpropagationStore Double] -> (NNOutput Double, [BackpropagationStore Double])
forwardPass (layer : layers) x backpropStores =
  let u = weights layer LA.<> x
      f = activationFun layer
      z = f u
      store = BackpropagationStore {prevInputX = x, activationFunResult = z}
   in forwardPass layers z $ store : backpropStores
forwardPass [] x backpropAcc = (x, backpropAcc)

-- Nice shorthands for loss functions
loss :: Loss -> Matrix Double -> Matrix Double -> Double
loss = getLoss

loss' :: Loss -> Matrix Double -> Matrix Double -> Matrix Double
loss' = getLoss'

activationFun :: Layer a -> Matrix Double -> Matrix Double
activationFun layer = getActivation $ activation layer

activationFun' :: Layer a -> Matrix Double -> Matrix Double
activationFun' layer = getActivation' $ activation layer

 

-- -- lossGrad w.r.t. current layer
-- backward :: NeuralNetwork Double -> [BackpropagationStore Double] -> Matrix Double -> (Matrix Double, [Gradients Double])
-- backward (layer : layers) (store : backpropStores) lossGrad =
--   let 
--     dY = activationFun' layer $ prevInputX store  * lossGrad 
--     dW = linearW' inp dY
--     dB = bias' dY
--     dX = linearX' w dY
--     grads = Gradients
--       { dbGradient = lossGrad,
--       dwGradient = weights layer
--       }
--     nextLayerLossGrad = lossGrad
--    in backward layers backpropStores lossGrad
-- backward _ _ lossGrad = (lossGrad, [])

-- where
--   singlePass layer = weights layer La.<> x -- - biases layer
--   f = getActivation (activation layer)


-- foo :: Layer a ->
foo :: Layer Double -> Matrix Double
foo layer = getActivation (activation layer) (biases layer)