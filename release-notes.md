### Added

- The code of the book's Chapters 6 and 7: `corr2d`, `browser_examples`, which cuts the number of training examples in the browser, and shims for the GPU functions (`cpu`, `gpu`, `num_gpus`, `try_gpu`, `try_all_gpus`), so the book's code runs unchanged: a browser has no GPU.
- `nn`: `Module.add_module`, `children`, `named_parameters`, `state_dict`, `load_state_dict` and `apply`; `Conv2d` and `LazyConv2d`, `MaxPool2d` and `AvgPool2d`, and `Sigmoid`, all differentiable by autograd; `init` with `normal_`, `uniform_`, `constant_`, `zeros_`, `ones_` and `xavier_uniform_`.

### Changed

- The LeNet notebook (7.6) trains on 10,000 images outside the browser, not the book's 60,000, so that it runs in minutes.
