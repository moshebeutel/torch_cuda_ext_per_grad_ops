# torch_cuda_ext_per_grad_ops
This repo contains implementation of pytorch cuda extension to some per gradient operations needed for differential privacy algorithms.  


## Task Overview
Given  neural network parameters, the gradients are calculated at each batch for the whole batch. For  the pupuse of differential privacy we would like to preform some per gradients operations:
1. Compute the **mean** of gradient along the batch axis.
2. Compute the **standard deviation** of gradients along the batch axis.
3. Compute the **median** of gradients along the batch axis.
4. Remove gradient **outliars**  along the batch axis.
5. Compute a **weighted sum** of the gradients along the batch axis where the weights are the probability density of the gradient value by the Gaussian defined by the mean and standard deviation calculated in 1. and 2. 