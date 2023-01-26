# CUDA Project - Implement `Pytorch` Missing Per-Sample-Gradient Operations


## Overview
`Pytorch` is a popular machine learning software framework. It handles all neural networks training and inference pipeline: 
 - Loading The Data
 - Data Preprocessing
 - Feed The data through the neural network
 - Computing the loss function w.r.t the desired labels
 - Compute the loss function gradient at each neuron needed for the learning.
 - Apply weight change according to loss gradients.

All the above can be done **on the GPU** very easyly in `Pytorch`. Moreover, all computation is done in data batches which is desired in most aspects - except *Differential Privacy* calculations.

## Differential Privacy
Differential Privacy is a mathematical promise that to some desired probability  a single data item participation in training process will not change the AI model predictions too much.

Usualy this is achieved using random manipulations on the gradients: additive Gaussian noise, random gradient ignore, manipulation using the gradient distribution etc.

In addition it is desired to continuosly track the privacy budget - the probability of the privacy promise which is changed during training.

Thus, Differential Privacy uses the the gradients calculated per example and not on the entire batch while the taraining process feeding the data forward, calculating the loss gradients and optimizing the weights backward is done in batches. 

Luckily, the per sample gradients are available using existing differential privacy libraries but other manipulations, among them those that are currently researched ofcourse,  are missing.

## Task Overview
Given  neural network parameters, the gradients are calculated at each batch for the whole batch. For  the pupuse of differential privacy we would like to preform some per sample gradients operations:
1. Compute the **mean** of gradient along the batch axis.
2. Compute the **standard deviation** of gradients along the batch axis.
3. Compute the **median** of gradients along the batch axis.
4. Remove gradient **outliars**  along the batch axis.
5. Compute a **weighted sum** of the gradients along the batch axis where the weights are the probability density of the gradient value by the Gaussian defined by the mean and standard deviation calculated in 1. and 2. 
6. **Clip** each gradient before taking the mean\median. 
7. Take gradient **sign** instead of its value. 
8. Get the per-sample-gradients and batch gradient for the same forward pass function.

## Future Plan
`Pytorch` contain the ability to be extended by the users. This is done using cpp extensions and CUDA integration. This allows to integrate  our implementd features into `Pytorch` as  built-in functions. 