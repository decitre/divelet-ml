### Added

- The code of the book's Chapter 12 (optimization algorithms), in `divelet_ml.optim`: `SGD` with `momentum`, and `Adagrad`, `RMSprop`, `Adadelta` and `Adam`, which update the parameter arrays in place with `step(grads)`; their `param_groups`, where changing a group's `'lr'` changes the next step; and `lr_scheduler.MultiStepLR` with `step()` and `get_last_lr()`.
- The helpers that Chapter 12's sections save for each other: `train_2d`, `show_trace_2d`, `annotate`, `Timer`, `Animator`, `Accumulator`, `linreg`, `squared_loss`, `evaluate_loss`, `accuracy`, `evaluate_accuracy_gpu`, `load_array`, `get_data_ch11`, `train_ch11` and `train_concise_ch11`. The adapted optimizer functions take the gradients as an argument, `trainer_fn(params, grads, states, hyperparams)`, in place of reading `p.grad`.
- `airfoil_self_noise()`, the NASA airfoil self-noise data set (UCI repository, CC BY 4.0) in the book's copy: read from the site in JupyterLite, downloaded and checked by its SHA-1 hash elsewhere.
- `load_data_fashion_mnist(batch_size, resize=None, root='../data', num_train=None, num_val=None)`, the book's data loader; `num_train` and `num_val` keep only the first examples (not in the book).
- `nn.MSELoss(reduction='mean' | 'sum' | 'none')`, as in PyTorch.

### Changed

- `optim.SGD(params, lr, momentum=0, weight_decay=0)` takes `momentum` as its third argument, as PyTorch's does; the third argument used to be `weight_decay`, so pass it by keyword. The optimizer's `weight_decays` attribute is gone: the weight decay of each parameter group is in `param_groups`.
