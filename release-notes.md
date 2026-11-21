### Added

- The code of the book's Chapter 8: `nn.BatchNorm1d`, `nn.BatchNorm2d` and their lazy versions `nn.LazyBatchNorm1d` and `nn.LazyBatchNorm2d` (batch statistics in training mode, running statistics in evaluation mode); `nn.AdaptiveAvgPool2d((1, 1))`; the `groups` argument of `nn.conv2d`, `nn.Conv2d` and `nn.LazyConv2d`; and `ResNeXtBlock`.
- `init_cnn` and `Classifier.layer_summary`, which the book saves from its earlier chapters.
- `nn.Sequential.add_module(name, module)`, which appends a named module, as in PyTorch.
- A module can list `buffers`: arrays that are state and not parameters, such as the running statistics of batch normalization. `parameters()` and `named_parameters()` leave them out, and `state_dict()` includes them.

### Changed

- `nn.Sequential[i]` raises `IndexError` when `i` is out of range; it used to wrap around.
