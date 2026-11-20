### Added

- The code of the book's Chapter 9: `Vocab`, `TimeMachine` (it reads the book's copy of *The Time Machine*, from the site in JupyterLite) and `time_machine_text`; the recurrent language models `RNNScratch`, `RNNLMScratch`, `RNN` and `RNNLM`, with `predict`; and `check_len`.
- `nn.RNN`: a one-layer tanh recurrent network with `torch.nn.RNN`'s interface, differentiable by autograd.

### Fixed

- `Trainer(..., gradient_clip_val=...)` called a `clip_gradients` method that did not exist; it clips the gradients to their joint L2 norm now.
