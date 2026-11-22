"""The recurrent language models of Chapter 9.

Adapted from `d2l/torch.py` in Dive into Deep Learning (https://github.com/d2l-ai/d2l-en, commit
23d7a5aecceee57d1292c56e90cce307f183bb0a), Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
Gradients come from autograd, so the models compute with autograd's NumPy.
"""

import autograd.numpy as anp
import numpy as np

from divelet_ml import nn
from divelet_ml.training import Classifier, Module
from divelet_ml.utils import numpy


class RNNScratch(Module):
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = (np.random.randn(num_inputs, num_hiddens) * sigma).astype(np.float32)
        self.W_hh = (np.random.randn(num_hiddens, num_hiddens) * sigma).astype(np.float32)
        self.b_h = np.zeros(num_hiddens, np.float32)

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = np.zeros((inputs.shape[1], self.num_hiddens), np.float32)
        else:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = anp.tanh(anp.matmul(X, self.W_xh) + anp.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state


class RNNLMScratch(Classifier):
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = (np.random.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma).astype(np.float32)
        self.b_q = np.zeros(self.vocab_size, np.float32)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', anp.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', anp.exp(l), train=False)

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        return np.eye(self.vocab_size, dtype=np.float32)[X.T]

    def output_layer(self, rnn_outputs):
        outputs = [anp.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return anp.stack(outputs, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = np.array([[outputs[-1]]])
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(numpy(Y).argmax(axis=2).reshape(-1)[0]))
        return ''.join([vocab.idx_to_token[i] for i in outputs])


class RNN(Module):
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)


class RNNLM(RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, hiddens):
        return anp.swapaxes(self.linear(hiddens), 0, 1)


class GRU(RNN):
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers, dropout=dropout)
