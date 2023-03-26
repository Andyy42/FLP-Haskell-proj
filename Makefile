NAME=flp22-fun
BUILD_DIR=build

install:
	cabal v1-install

run_cabal:
	cabal run $(NAME) -- configs/iris_small.conf --experiment 0

build_dir:
	mkdir -p $(BUILD_DIR) 

build: build_dir
	cd src/ && ghc -O3 -Wall -dynamic Main.hs -o $(NAME) && mv $(NAME) ../build/$(NAME)

run_iris_small: build
	$(BUILD_DIR)/$(NAME) --config configs/iris_small.conf --verbose --experiment 0

run_iris_tanh: build
	$(BUILD_DIR)/$(NAME) --config configs/iris_tanh.conf --verbose --experiment 0

run_iris_id: build
	$(BUILD_DIR)/$(NAME) --config configs/iris_id.conf --verbose --experiment 0

run_iris_relu: build
	$(BUILD_DIR)/$(NAME) --config configs/iris_relu.conf --verbose --experiment 0

run_iris_big: build
	$(BUILD_DIR)/$(NAME) --config configs/iris_big.conf --verbose --experiment 0

run_mnist_small: build
	$(BUILD_DIR)/$(NAME) --config configs/mnist_small.conf --verbose --experiment 2

run_mnist_big: build
	$(BUILD_DIR)/$(NAME) --config configs/mnist_big.conf --verbose --experiment 2

run_iris_all: run_iris_small run_iris_tanh run_iris_id run_iris_relu run_iris_big
run_mnist_all: run_mnist_small run_mnist_big 
run_all: run_iris_all run_mnist_all

# Usage: neuralNetworksProj --config FILE [OPTIONS...]
#   -h         --help               Print help and exit
#   -c FILE    --config=FILE        Required path to configuration file
#   -v         --verbose            Verbose mode reads config, create neural network and prints results
#   -e NUMBER  --experiment=NUMBER  Enable experiment 0, 1, 2, or 3
	

clean:
	rm $(BUILD_DIR)/$(NAME) && cd src/ && rm *.o *.hi


