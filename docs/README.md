
#### **Table with different types of matrix multiplications  and their results**

| **Types**  | **Scalar**                                           | **Vector**                                                          | **Matrix**                                       |   |
|------------|------------------------------------------------------|---------------------------------------------------------------------|--------------------------------------------------|---|
| **Scalar** | $\frac{\partial{y}}{\partial{x}}$ = scalar              | $\frac{\partial{\mathbf{y}}}{\partial{x}}$= **column** vector              | $\frac{\partial{\mathbf{Y}}}{\partial{x}}$= matrix |   |
| **Vector** | $\frac{\partial{y}}{\partial{\mathbf{x}}}$= **row** vector | $\frac{\partial{\mathbf{y}}}{\partial{\mathbf{x}}}$ = matrix           |                                                  |   |
| **Matrix** | $\frac{\partial{y}}{\partial{X}}$= matrix              | $\frac{\partial{\mathbf{y}}}{\partial{\mathbf{X}}}$ = 3rd order tensor |                                                  |   |\

### Forward propagation

All vectors are column vectors




### Single layer

Single layer of fully-connected linear neural network consist of ${M}^{(\ell)}$ neurons where the input vector $\mathbf{z^{(\ell-1)}}$ has ${M}^{(\ell-1)}$ neurons and has total length $L$. Neurons from previous layer are fully-connected with current layer neurons by weighted connections with weight matrix $\mathbf{W^{(\ell)}}$ which has dimension of ${M}^{(\ell-1)}\times{M}^{(\ell)}$ where rows dim ${M}^{(\ell-1)}$ also represents number of connections to single neuron. We also add bias ${b_n^{(\ell)}}$ to each neuron $n$ connection with bias vector $\mathbf{b}^{(\ell)}$. Vector of values $\mathbf{u}^{(\ell)}$ of neurons in current layer can be denoted as:
$$
\mathbf{u^{(\ell)}} = \mathbf{z^{(\ell-1)}}\mathbf{W^{(\ell)}} + \mathbf{b^{(\ell)}}
$$

With corresponding derivations:

$$
\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{z}^{(\ell-1)}} = \mathbf{W^{(\ell)}}
$$

$$
\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{W}^{(\ell)}} = \bigr(\mathbf{z^{(\ell-1)}}\bigr)^T \cdot \_ \space\space\text{(this one is bit more complex, we'll get to it later)}
$$

$$
\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}} = \mathbf{I}
$$

where $\mathbf{I}$ is the identity matrix.

Let us denote $f$ as our activation function then one layer of neural network with nonlinearity would be written as: 
$$
\mathbf{z^{(\ell)}} =f(\mathbf{u}^{(\ell)})= f(\mathbf{z}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)})
$$

where  $\mathbf{z^{(\ell)}}$ is the final output from one layer after the activation function, which is then propagated to next layer. The derivation w.r.t. to the output of linear layer would be:
$$
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}} = \frac{\partial f(\mathbf{u}^{(\ell)})}{\partial\mathbf{u}^{(\ell)}} = f'(\mathbf{u}^{(\ell)})
$$

For further use we'll denote the derivation of neorons output w.r.t to output of linear layer as $f'(\mathbf{u}^{(\ell)})=\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}}=\mathbf{F}^{(\ell)}$.

### Loss function

The  last layer's output is used in calculating the overall error of our neural network with loss function as:

$$
\text{Error}=E(\mathbf{z}^{(L)}, \mathbf{t}) \in \R
$$

Note that result is a single digit (scalar). This will be usefull in the future.

### Derivations of neural network

Now things get bit more complicated with derivations of $\mathbf{z}^{(\ell)}$ w.r.t. to 
matrix $\mathbf{W}^{(\ell)}$ and vectors
$\mathbf{z}^{(\ell-1)}$, $\mathbf{b}^{(\ell)}$. By looking at [Matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus) at Wikipedia we can see that it might be quite an intimidating task.

Anyway, let's start with derivation of two vectors. Result should be the *differential* or *Jacobian* matrix. For some arbitrary $\frac{\partial{\mathbf{y}}}{\partial{\mathbf{x}}}$ the matrix is simply made of all combinations of $\frac{\partial{{y}_i}}{\partial{{x}_j}}$ which can be transleted as **how does a tiny nudge in $x_j$ influences the value of $y_i$**?

#### **Derivation w.r.t. bias**

So in $\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}}$  that means how does a tiny nudge to $b_j^{(\ell)}$ influences value of $z_i^{(\ell)}$. But becuase each bias $i$ influences only neuron $i$ then all 
$\frac{\partial{z_i}^{(\ell)}}{\partial{b_j}^{(\ell)}}$ where are $i\neq{j}$ are 0. For $i=j$ we use chain rule:
$$
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}} =
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}}\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}} = 
f'(\mathbf{u^{(\ell)}})\mathbf{I} = \mathbf{F}^{(\ell)}
$$

Notice that we can use the same trick for activation function, some of them (not you softmax!) use as an input the output from single linear layer. In other words, tiny nudge in ${{u}_i^{(\ell)}}$ will make tiny change in ${{z}_j^{(\ell)}}$ if $i=j$ otherwise the ${{u}_i^{(\ell)}}$ does not influence ${{z}_j^{(\ell)}}$. For some activation functions (*component-wise nonlinearities*) result can be simplified to a diagonal matrix. The resulting equation is:

$$
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}}=
\mathbf{F}^{(\ell)}
$$

#### **Derivation w.r.t. previous layer output $\mathbf{z}^{(\ell-1)}$**

Expanding the derivation by chain rule we get:
$$
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{z}^{(\ell-1)}} =
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}}\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{z}^{(\ell-1)}}
$$
Now let's look closely at term $\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{z}^{(\ell-1)}}$ and expand it:

$$
\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{z}^{(\ell-1)}}=
\frac{\partial}{{\partial\mathbf{z}^{(\ell-1)}}}\Bigl(
\mathbf{W^{(\ell)}}\mathbf{z^{(\ell-1)}} + \mathbf{b^{(\ell)}}\Bigr)
=
\frac{\partial \bigl(\mathbf{W^{(\ell)}}\mathbf{z^{(\ell-1)}}\bigr)}{{\partial\mathbf{z}^{(\ell-1)}}} + 0 = \mathbf{W^{(\ell)}}
$$

Putting it together we got
$$
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{z}^{(\ell-1)}}= f'(\mathbf{u^{(\ell)}}) \mathbf{W^{(\ell)}}= 
\mathbf{F}^{(\ell)}\mathbf{W^{(\ell)}}
$$

Also note that this equations shows how is single output of current layer $\ell$ is influenced by changes in the previous layer $\ell-1$:
$$
\frac{\partial\mathbf{z}^{(\ell)}}{\partial{z_i}^{(\ell-1)}}=
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}}\frac{\partial\mathbf{u}^{(\ell)}}{\partial{z_i}^{(\ell-1)}}=
{f'({\mathbf{u}^{(\ell)}}) {\mathbf{w}_{i}^{(\ell)}}} = 
{{\mathbf{F}^{(\ell)}} {\mathbf{w}_{i}^{(\ell)}}}

$$
$$
\frac{\partial{z}_j^{(\ell)}}{\partial{\mathbf{z}}^{(\ell-1)}}=
\frac{\partial{z_j}^{(\ell)}}{\partial{u_j}^{(\ell)}}\frac{\partial{u_j}^{(\ell)}}{\partial{\mathbf{z}}^{(\ell-1)}}=
{f'({{u}^{(\ell)}}) {\mathbf{w}_{j}^{(\ell)}}}
$$


#### **Derivation w.r.t. weights $\mathbf{W}^{(\ell)}$**

This is bit more problematic becuase we cannot simply do $\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{W}^{(\ell)}}$ and make derivation of vector w.r.t. matrix as that would result in 3rd order tensor, but that's ugly. We know that not all weights influences all outputs (lots of 0 in this 3rd order tensor) but rather single column influences one neuron! So we'll have to use indexes and take columns of our weight matrix a scalar out of vector of outputs to get:
$$
\frac{\partial{z_j}^{(\ell)}}{\partial{\mathbf{w}_{j}}^{(\ell)}}=
\frac{\partial{z_j}^{(\ell)}}{\partial{{u}_{j}}^{(\ell)}}
\frac{\partial{u_j}^{(\ell)}}{\partial{\mathbf{w}_{j}}^{(\ell)}}
$$

where we also used *chain rule*. The $\mathbf{w}_{j}$ is now column vector and ${z_j}^{(\ell)}$, ${u_j}^{(\ell)}$ are scalars. Lets calculate the derivatives separately:

$$
\frac{\partial{z_j}^{(\ell)}}{\partial{{u}_{j}}^{(\ell)}} = f'(u_j^{(\ell)}) \\
\frac{\partial{u_j}^{(\ell)}}{\partial{\mathbf{w}_{j}}^{(\ell)}} =
\bigl(\mathbf{w}_{j}^{(\ell)}\bigr)^T\mathbf{z}^{(\ell-1)} + b_j^{(\ell)} = \mathbf{z}_i^{(\ell-1)}
$$

where $\mathbf{z}_i^{(\ell-1)}$ is a **row vector** with $z_i^{(\ell-1)}$ at index $j$ in $\mathbf{z}_i^{(\ell-1)}$. Putting it all together

$$
\frac{\partial{z_j}^{(\ell)}}{\partial{\mathbf{w}_{j}}^{(\ell)}}=
f'(u_j^{(\ell)})\cdot\mathbf{z}_i^{(\ell-1)}
$$

We would get the similar same result from

$$
\frac{\partial{\mathbf{z}}^{(\ell)}}{\partial{{w}_{ij}}^{(\ell)}}=
f'(u_j^{(\ell)})\cdot\mathbf{z}_i^{(\ell-1)}
$$

Except the $\mathbf{z}_i^{(\ell-1)}$ is a **column vector**.

How to get to the derivation of tensor:
1. Now for $\frac{\partial{\mathbf{u}}^{(\ell)}}{\partial{\mathbf{{w}_j}}^{(\ell)}}$ with a fixed single output $j$ the result is a $i\times{j'}$ (row,col) matrix with gradients only on the $j$-th column, because the $i$-th input node can only influence the $j$-th output node by $w_{ij'}$ weight where the $j'$-th index matches the single output node index $j=j'$ (otherwise it's a weight of some different output node we don't care now).

2. And for $\frac{\partial{\mathbf{u}}^{(\ell)}}{\partial{\mathbf{{w}_i}}^{(\ell)}}$ with fixed single input neuron $i$ the result is a $j\times{j'}$ matrix with gradients only diagonal, this should be clear because the $i$-th node can be influenced only by $w_{ij'}$ weight where $j=j'$ and with the gradient equal to ${z}_i$ (because the $i$ is fixed).

3. For the 3rd order tensor $\frac{\partial{\mathbf{u}}^{(\ell)}}{\partial{\mathbf{{W}}}^{(\ell)}}$ we have to combine  the above. We'll introduce one more index $k$ for ${u_k}^{(\ell)}$. So our resulting tensor, let's call it $T$, will have indexes as $\mathbf{T}_{kij}$ ($k$=row,$i$=col,$j$=depth) and the "face" of the tensor will be $\mathbf{T}_{ki}$ (which is matrix). So we see that the "face" has fixed $k$ but this means that the output node is also fixed! We now now that the "face" is the matrix $i\times{j'}$ from 1. but inverse so $k\times{i}$ matrix with values only on single row where $k=j'$ and the other dimension $\mathbf{T}_{kj}$ is the diagonal matrix from 2. The final tensor is sort of diagonal tensor with the same values $z_i$ on the diagonal and different values of $z$ in a row and 0 otherwise. Notice that the overall information in the tensor can be reduced to single vector!

For right now we'll leave it here since $\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{W}^{(\ell)}}$, especially the $\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{W}^{(\ell)}}$ part, is more complicated to solve and a neat trick will come later with multiplication of vector and diagonal 3rd order tensor which will simplify it a lot. 

So now we have all the pieces of the puzzle we need for backpropagation!

### Backpropagation


How does current layer weights influence the overall error 
$\frac{\partial{E}}{\partial\mathbf{W}^{(\ell)}}$? We know that the overall error depends directly on output of previous layer, which is influenced by previous layer and so on. This sounds like *chain rule* might come to rescue: 


$$
\frac{\partial{E}}{\partial\mathbf{W}^{(\ell)}} = 
\frac{\partial{E}}{\partial\mathbf{z}^{(L)}} 
\frac{\partial\mathbf{z}^{(L)}}{\partial\mathbf{z}^{(L-1)}}\dots
\frac{{\partial\mathbf{z}^{(\ell+1)}}}{\partial\mathbf{z}^{(\ell)}}
\frac{{\partial\mathbf{z}^{(\ell)}}}{\partial\mathbf{W}^{(\ell)}}
$$


We can simplify some parts of the equation
$$
\delta^{(\ell)}=
\frac{\partial{E}}{\partial\mathbf{z}^{(\ell)}} = 
\frac{\partial{E}}{\partial\mathbf{z}^{(L)}}\cdot
\frac{\partial\mathbf{z}^{(L)}}{\partial\mathbf{z}^{(L-1)}}\dots
\frac{{\partial\mathbf{z}^{(\ell+1)}}}{\partial\mathbf{z}^{(\ell)}}
$$
$$
\frac{\partial{E}}{\partial\mathbf{W}^{(\ell)}} = 
\delta^{(\ell)}
\frac{{\partial\mathbf{z}^{(\ell
)}}}{\partial\mathbf{W}^{(\ell)}}
$$

Where the $\delta^{(\ell)}$ is a **row vector** (this will be quite usefull soon!). We can calculate current $\delta^{(\ell)}$ from the next one $\delta^{(\ell+1)}$ recursively as

$$
\delta^{(\ell)} = 
\delta^{(\ell+1)}
\frac{{\partial\mathbf{z}^{(\ell+1)}}}{\partial\mathbf{z}^{(\ell)}} = 
\delta^{(\ell+1)} \mathbf{F}^{(\ell+1)} \mathbf{W^{(\ell+1)}}
$$

Now let's get back to the dreaded 3rd order tensor $\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{W}^{(\ell)}}$ and see what we can do with it by expanding it in $\frac{\partial{E}}{\partial\mathbf{z}^{(\ell)}}$ as

$$
\frac{\partial{E}}{\partial\mathbf{W}^{(\ell)}} = 
\delta^{(\ell)}
\frac{{\partial\mathbf{z}^{(\ell)}}}{\partial\mathbf{u}^{(\ell)}} 
\frac{{\partial\mathbf{u}^{(\ell)}}}{\partial\mathbf{W}^{(\ell)}}
$$

Notice that the $\delta^{(\ell)}\frac{{\partial\mathbf{z}^{(\ell)}}}{\partial\mathbf{u}^{(\ell)}}$ is a matrix multiplication of a row vector and a matrix so again **row vector** after all let's denote this vector as $\xi^{(\ell)}=\delta^{(\ell)}\frac{{\partial\mathbf{z}^{(\ell)}}}{\partial\mathbf{u}^{(\ell)}}$. If we multiply it with our tensor $\frac{{\partial\mathbf{u}^{(\ell)}}}{\partial\mathbf{W}^{(\ell)}}$ it'd collapse to a matrix. As was explained before, the tensor with dimensions $kij$ has $z$ values in rows (basicly there's whole $\mathbf{z}^{(\ell)}$ vector) and if we look at it from front (so in $ki$ dimension) where the row is same as value of $k$ for that sub-matrix  of the tensor.
And same values of $z_i$ on a diagonal in $kj$ dimension. Now if you imagine (or code it in Python) multyplying such tensor by vector $\xi^{(\ell)}\frac{{\partial\mathbf{u}^{(\ell)}}}{\partial\mathbf{W}^{(\ell)}}$ is the same as taking the $\mathbf{z^{(\ell)}}$ (which is duplicated in every $j=k$ row of the tensor along $i$-th dimension) and do a simple matrix multiplication $\bigl(\mathbf{z^{(\ell)}}\bigr)^T\xi^{(\ell)}= \bigl(\xi^{(\ell)}\bigr)^T\mathbf{z^{(\ell)}}$ where $\xi^{(\ell)}=\delta^{(\ell)}\frac{{\partial\mathbf{z}^{(\ell)}}}{\partial\mathbf{u}^{(\ell)}}=\delta^{(\ell)}\mathbf{F^{(\ell)}}$



To summarize, during *backpropagation* we want to update every variable by a reasonable amount by using *gradient descent*
so we minimize the overall value of the total error $E$ obtained from the loss function.
So for each layer's weight matrix $\mathbf{W}^{(\ell)}$ and bias vector $\mathbf{b}^{(\ell)}$ we want to calculate gradient w.r.t error $E$ as:

$$
\frac{\partial{E}}{\partial\mathbf{W}^{(\ell)}} = 
\delta^{(\ell)}
\frac{{\partial\mathbf{z}^{(\ell
)}}}{\partial\mathbf{W}^{(\ell)}} =
\bigl(\mathbf{z}^{(\ell-1)}\bigr)^T
\bigl(\delta^{(\ell)}
\mathbf{F}^{(\ell)}\bigr)
$$
$$
\frac{\partial{E}}{\partial\mathbf{b}^{(\ell)}} = 
\delta^{(\ell)}
\frac{{\partial\mathbf{z}^{(\ell
)}}}{\partial\mathbf{b}^{(\ell)}} =
\delta^{(\ell)}
\frac{\partial\mathbf{z}^{(\ell)}}{\partial\mathbf{u}^{(\ell)}}\frac{\partial\mathbf{u}^{(\ell)}}{\partial\mathbf{b}^{(\ell)}} = 
\delta^{(\ell)}
\mathbf{F}^{(\ell)}
\mathbf{I}
$$
$$
\delta^{(\ell)}=
\delta^{(\ell+1)}
\frac{{\partial\mathbf{z}^{(\ell+1)}}}{\partial\mathbf{z}^{(\ell)}} = 
\delta^{(\ell+1)} \mathbf{F}^{(\ell+1)} \mathbf{W^{(\ell+1)}}
$$
$$
\delta^{(L)}=
\frac{\partial{E}}{\partial\mathbf{z}^{(L)}}
$$

**Caveats:** Implementation detail for: $\delta^{(\ell+1)}\mathbf{F}^{(\ell+1)}$ can be computed as the *Hadamard product* (pair-wise multiplication)

### Gradient descent

Gradient descent uses simple update rule to update neural network parameters $\theta=\{\mathbf{W}^{(\ell)}, \mathbf{b}^{(\ell)}\}$ as

$$
\theta :=\theta-\gamma\nabla_\theta{J(\theta)}
$$

Where $\nabla_\theta{J(\theta)}$ is the gradient of the parameter calculated as partial derivations from previous section and $\gamma$ is the *learning rate*. We iterate this forward and backward propagation with parameter update until we obtain optimal train and validation loss.

### Ativations

#### **Sigmoid**

Related to Logistic Regression. For single-label/multi-label binary classification.

$$
\sigma(x) = \frac{1} {1 + e^{-x}}
$$
$$
\sigma(x)' =  \sigma(x)(1-\sigma(x))
$$

#### **Tanh**

$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}}
$$
$$
tanh(x)' = 1 - tanh^2(x)
$$

#### **Softmax**

For multi-class single label classification.

$$
S_i = \frac{e^{x_{i}}}{\sum_{j=1}^K e^{x_{j}}} \ \ \ for\ i=1,2,\dots,K \\
\partial_i{S_j} =
\begin{cases}
S_i(1-S_i) & i=j\\
-S_iS_j & i\neq{j}\\
\end{cases}
$$

#### **ReLU**

$$
ReLU(z) = max(0, z) \\
ReLU(x)' =
\begin{cases}
1 & \text{if} x > 0 \\
0 & \text{otherwise}
\end{cases}
$$

### Loss functions

#### **Mean Squared Error(MSE)**
For regression tasks. Below targets $\mathbf{t}$ and predictions $\mathbf{y}$ are $D$ dimensional vectors, and $t_i$ denotes the value on the $i$-th dimension of $\mathbf{t}$.

$$
E_{MSE}=\frac{1}{D}\sum_{i=1}^{D}(x_i-y_i)^2 \\
\frac{\partial{E_{MSE}}}{\partial{\mathbf{y}}} = (y_i - t_i)_i
$$

#### **Cross Entropy**
For classification tasks. The $i$-th dimension of vectors $\mathbf{t}$ and $\mathbf{y}$ represents *class* in our classification task.

$$
E_{CE}=-\sum_{i=1}^Dt_{i}\log(y_{i}) \\
\frac{\partial{E_{CE}'}}{\partial{\mathbf{y}}}=\Bigl(-\frac{t_k}{y_{i}}\Bigr)_k
$$













