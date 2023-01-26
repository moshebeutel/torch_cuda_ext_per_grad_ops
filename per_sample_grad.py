
from functorch import make_functional_with_buffers, vmap, grad

def compute_loss_stateless_model_func(net, criterion):
    fmodel, params, buffers = make_functional_with_buffers(net)
    def compute_loss_stateless_model (params, buffers, sample, target):

        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = fmodel(params, buffers, batch) 
        loss = criterion(predictions, targets)
        # print("loss", loss.shape)
        # running_loss += float(loss)
        return loss
    
    return compute_loss_stateless_model, params, buffers


def apply_along_axis(func1d, axis: int, arr, *args, **kwargs):
  num_dims = arr.ndim
  # axis = _canonicalize_axis(axis, num_dims)
  func = lambda arr: func1d(arr, *args, **kwargs)
  for i in range(1, num_dims - axis):
    func = vmap(func, in_dims=i, out_dims=-1)
  for i in range(axis):
    func = vmap(func, in_dims=0, out_dims=0)
  return func(arr)


def get_per_sample_grads(net, inputs, labels, criterion):
    compute_loss_stateless_model, params, buffers = compute_loss_stateless_model_func(net, criterion)    
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs, labels)
    return ft_per_sample_grads