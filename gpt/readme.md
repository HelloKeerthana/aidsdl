<h2>bigram model<h2> 

batch size = 65
block size / context window = 256
6 layers, 6 heads
vector dim = 65
learning rate 0.0003 - on such data attention dont do good with high learning rate
dropout 0.2
max iterations 5k
eval intervals 500, iters 200

<h2>gpt model<h2> [not the same, diff number of layers and hyperparams]
batch size = 32
block size / context window = 8
1 layer, 1 head - self attention
vector dim = 32
learning rate 0.001
max iterations 3k
eval iters 300

dataset - shakesphere
dont run model unless good gpu - gpt - 10M corpus
bigram can run on cpu

credit - andrej karpathy
