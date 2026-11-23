"""An example of `nbbuild`: Section 10.4 (*Bidirectional Recurrent Neural Networks*) rebuilt from the book's sources.

    D2L_SRC=d2l-en python tools/d2l_convert/example_bi_rnn.py [OUT_DIR]

It writes `10.04_bi-rnn.ipynb` to OUT_DIR (default: a temporary folder). `tests/test_converter.py` checks that it equals `content/10_recurrent-modern/10.04_bi-rnn.ipynb`
cell by cell, except for the cell ids, which are random. The notebooks are edited by hand after they are built, so this is how a section starts, not how it is kept.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbbuild import Chapter, LIB, NUMPY  # noqa: E402

SECTIONS = [("lstm", "Long Short-Term Memory (LSTM)"), ("gru", "Gated Recurrent Units (GRU)"), ("deep-rnn", "Deep Recurrent Neural Networks"),
            ("bi-rnn", "Bidirectional Recurrent Neural Networks"), ("machine-translation-and-dataset", "Machine Translation and the Dataset"),
            ("encoder-decoder", "The Encoder--Decoder Architecture"), ("seq2seq", "Sequence-to-Sequence Learning for Machine Translation"), ("beam-search", "Beam Search")]
IMPORTS = "from divelet_ml import nn\nimport autograd.numpy as np\n\nimport divelet_ml as d2l"


def build(out_dir, adapted=tuple(range(1, 9))):  # all of Chapter 10 exists, so 10.4 links forward to 10.5
    chapter = Chapter(10, "chapter_recurrent-modern", out_dir, SECTIONS, adapted,
                      previous="Previous: [Section 9.7: Backpropagation Through Time](../09_recurrent-neural-networks/09.07_bptt.ipynb)")
    return chapter.build("bi-rnn", {
        1: IMPORTS,
        7: """class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2""",
        5: """@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(inputs[::-1], b_H)
    outputs = [np.concatenate((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)""",
    }, [NUMPY, LIB, "`reversed(inputs)` becomes `inputs[::-1]`: PyTorch's tensors can be reversed with `reversed`, but NumPy's arrays give an iterator without a shape. `d2l.concat` becomes `np.concatenate`.",
        "divelet-ml's `nn.GRU` is not bidirectional, so the `BiGRU` class is defined as in the book but cannot be built here; a cell after it runs `BiRNNScratch` instead (not in the book)."],
        added=[(8, "divelet-ml's `nn.GRU` has no `bidirectional` argument, so the `BiGRU` class above is defined as in the book but would raise an error if you built it. `BiRNNScratch` works: the cell below runs it on random inputs, and its outputs have twice as many features as the hidden state, `num_hiddens` doubled.",
               """bi_rnn = BiRNNScratch(num_inputs=4, num_hiddens=8)
outputs, (f_H, b_H) = bi_rnn(np.random.randn(5, 2, 4).astype(np.float32))
d2l.check_shape(outputs[0], (2, 16))  # (batch size, 2 * num_hiddens), for each of the 5 steps""")])


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()))
