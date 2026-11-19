"""Multiply-adds (MACs) of the Chapter 8 models, counted from the layer shapes of the book's PyTorch code.

    uv run python spikes/js-conv/count_macs.py

Counts convolutions and linear layers only, per image, forward; batch norm, activations, pooling and the
additions of residual connections are left out (small next to the convolutions). Input sizes are the book's:
`layer_summary` and training use the same size except VGG, which is summarized at full width and trained at a quarter.
The time estimates use the throughput measured by numpy_conv.py and index.html on one Mac (see the spike's PR).
"""
PYODIDE = 1.5e9    # MAC/s: NumPy convolution in Pyodide (middle case, forward)
WEBGPU = 150e9     # MAC/s: TensorFlow.js WebGPU, forward+backward counted as 3x forward (batch 8, middle case)
IMAGES, EPOCHS = 60000, 10

class Net:
    def __init__(self, c, h, w): self.c, self.h, self.w, self.macs = c, h, w, 0
    def conv(self, cout, k, s=1, p=0, groups=1):
        ho, wo = (self.h + 2 * p - k) // s + 1, (self.w + 2 * p - k) // s + 1
        self.macs += ho * wo * cout * (self.c // groups) * k * k
        self.c, self.h, self.w = cout, ho, wo
    def pool(self, k, s, p=0):
        self.h, self.w = (self.h + 2 * p - k) // s + 1, (self.w + 2 * p - k) // s + 1
    def gap(self): self.h = self.w = 1
    def linear(self, n):
        self.macs += self.c * self.h * self.w * n
        self.c, self.h, self.w = n, 1, 1

def stem7(n):  # ResNet's and DenseNet's first block
    n.conv(64, 7, 2, 3); n.pool(3, 2, 1)

def alexnet():
    n = Net(1, 224, 224)
    n.conv(96, 11, 4, 1); n.pool(3, 2); n.conv(256, 5, 1, 2); n.pool(3, 2)
    n.conv(384, 3, 1, 1); n.conv(384, 3, 1, 1); n.conv(256, 3, 1, 1); n.pool(3, 2)
    n.linear(4096); n.linear(4096); n.linear(10); return n

def vgg(arch):
    n = Net(1, 224, 224)
    for convs, c in arch:
        for _ in range(convs): n.conv(c, 3, 1, 1)
        n.pool(2, 2)
    n.linear(4096); n.linear(4096); n.linear(10); return n

def nin():
    n = Net(1, 224, 224)
    def blk(c, k, s, p): n.conv(c, k, s, p); n.conv(c, 1); n.conv(c, 1)
    blk(96, 11, 4, 0); n.pool(3, 2); blk(256, 5, 1, 2); n.pool(3, 2); blk(384, 3, 1, 1); n.pool(3, 2)
    blk(10, 3, 1, 1); n.gap(); return n

def googlenet():
    n = Net(1, 96, 96)
    def inc(c1, c2, c3, c4):
        cin, h, w = n.c, n.h, n.w
        # each branch starts from the block's input
        total = 0
        for chain in ([(c1, 1, 0)], [(c2[0], 1, 0), (c2[1], 3, 1)], [(c3[0], 1, 0), (c3[1], 5, 2)], [(c4, 1, 0)]):
            m = Net(cin, h, w)
            for cout, k, p in chain: m.conv(cout, k, 1, p)
            total += m.macs
        n.macs += total; n.c = c1 + c2[1] + c3[1] + c4
    n.conv(64, 7, 2, 3); n.pool(3, 2, 1); n.conv(64, 1); n.conv(192, 3, 1, 1); n.pool(3, 2, 1)
    inc(64, (96, 128), (16, 32), 32); inc(128, (128, 192), (32, 96), 64); n.pool(3, 2, 1)
    for a in [(192, (96, 208), (16, 48), 64), (160, (112, 224), (24, 64), 64), (128, (128, 256), (24, 64), 64),
              (112, (144, 288), (32, 64), 64), (256, (160, 320), (32, 128), 128)]: inc(*a)
    n.pool(3, 2, 1); inc(256, (160, 320), (32, 128), 128); inc(384, (192, 384), (48, 128), 128)
    n.gap(); n.linear(10); return n

def bnlenet():
    n = Net(1, 28, 28)
    n.conv(6, 5); n.pool(2, 2); n.conv(16, 5); n.pool(2, 2); n.linear(120); n.linear(84); n.linear(10); return n

def lenet():  # 7.6, for scale
    n = Net(1, 28, 28)
    n.conv(6, 5, 1, 2); n.pool(2, 2); n.conv(16, 5); n.pool(2, 2); n.linear(120); n.linear(84); n.linear(10); return n

def resnet18():
    n = Net(1, 96, 96); stem7(n)
    for i, (reps, c) in enumerate([(2, 64), (2, 128), (2, 256), (2, 512)]):
        for j in range(reps):
            down = j == 0 and i != 0
            cin, s = n.c, 2 if down else 1
            n.conv(c, 3, s, 1); n.conv(c, 3, 1, 1)
            if down:  # 1x1 on the block's input, at the output's size
                n.macs += n.h * n.w * c * cin
    n.gap(); n.linear(10); return n

def densenet(growth=32, arch=(4, 4, 4, 4)):
    n = Net(1, 96, 96); stem7(n)
    for i, convs in enumerate(arch):
        c = n.c
        for _ in range(convs):
            m = Net(c, n.h, n.w); m.conv(growth, 3, 1, 1); n.macs += m.macs; c += growth
        n.c = c
        if i != len(arch) - 1:
            n.conv(c // 2, 1); n.pool(2, 2)
    n.gap(); n.linear(10); return n

def regnetx32():
    n = Net(1, 96, 96); n.conv(32, 3, 2, 1)
    for depth, c, groups, bot_mul in [(4, 32, 16, 1), (6, 80, 16, 1)]:
        for i in range(depth):
            cin, s, bot = n.c, 2 if i == 0 else 1, int(round(c * bot_mul))
            n.conv(bot, 1); n.conv(bot, 3, s, 1, groups=bot // groups); n.conv(c, 1)
            if i == 0: n.macs += n.h * n.w * c * cin
    n.gap(); n.linear(10); return n

MODELS = [('7.6 LeNet (for scale)', lenet), ('8.1 AlexNet, 224', alexnet),
          ('8.2 VGG-11 full width, 224 (summary)', lambda: vgg(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))),
          ('8.2 VGG-11 quarter width, 224 (training)', lambda: vgg(((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)))),
          ('8.3 NiN, 224', nin), ('8.4 GoogLeNet, 96', googlenet), ('8.5 BN-LeNet, 28', bnlenet),
          ('8.6 ResNet-18, 96', resnet18), ('8.7 DenseNet, 96', densenet), ('8.8 RegNetX-32, 96', regnetx32)]

def fmt(s): return f'{s:.1f} s' if s < 120 else f'{s/60:.0f} min' if s < 7200 else f'{s/3600:.0f} h'

if __name__ == '__main__':
    # self-check against a hand count: a 3x3 conv, 2 -> 4 channels, same padding on 5x5, is 25*4*2*9 MACs
    m = Net(2, 5, 5); m.conv(4, 3, 1, 1); assert m.macs == 25 * 4 * 2 * 9
    print(f'{"model":42s} {"MMAC":>8s} {"1 image, Pyodide":>17s} {"epoch, Pyodide":>15s} {"epoch, WebGPU":>14s}')
    for name, f in MODELS:
        m = f().macs; one = m / PYODIDE; epoch = 3 * m * IMAGES
        print(f'{name:42s} {m/1e6:8.1f} {fmt(one):>17s} {fmt(epoch/PYODIDE):>15s} {fmt(epoch/WEBGPU):>14s}')
