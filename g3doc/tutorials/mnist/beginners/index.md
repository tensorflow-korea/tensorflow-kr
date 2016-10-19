# MNIST 초급

*이 튜토리얼은 머신러닝과 텐서플로우을 처음 접해본 독자들을 위해서 구성되어 있습니다. 만약 MNIST가 무엇인지 알고 소프트맥스(다변량 로지스틱) 회귀가 무엇인지 이미 알고 계신다면, [더 깊이 있는 튜토리얼](../pros/index.md)을 선호하실 것이라고 생각합니다. 튜토리얼을 진행하시기 전에, 자신의 머신에 [텐서플로우가 설치](../../../get_started/os_setup.md)되어 있는지 확인해주세요.*

프로그래밍을 어떻게 하는지 배울 때, 언제나 "Hello World."를 가장 먼저 출력해본다는 전통이 있습니다. 그것처럼, 머신러닝에선 MNIST를 가장 먼저 다룹니다.

MNIST는 간단한 컴퓨터 비전 데이터셋입니다. 이는 다음과 같이 손으로 쓰여진 이미지로 구성되어 있습니다.

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/MNIST.png">
</div>

또한, 이것은 그 숫자가 어떤 숫자인지 알려주는 각 이미지에 관한 라벨을 포함하고 있습니다. 예를 들어서, 위의 이미지의 경우에 라벨은 5, 0, 4, 그리고 1입니다.

이 튜토리얼에서는 모델이 이미지를 보고 어떤 숫자인지 예측하는 모델을 훈련시킬 것입니다. 우리의 목적은 가장 최신의 성능을 발휘하는 정교한 시스템을 만드는 것이 아닙니다.(물론 뒤에 가서 하게될 것입니다!) 대신에, 여기서는 텐서플로우에 발을 살짝만 담궈봅시다. 우리는 소프트맥스 회귀라고 불리는 아주 간단한 모델로 시작할 것입니다.

이 튜토리얼에 쓰이는 실제 코드는 매우 짧고, 흥미로운 일들은 단지 세 줄 안에 모두 일어납니다. 그러나 우리에게는 그 세 줄 안에 담겨있는 텐서플로우의 동작 원리와 머신러닝의 핵심 개념을 아는 것이 중요합니다. 때문에, 우리는 코드를 하나하나 주의 깊게 다룰 것입니다.

## MNIST 데이터셋

MNIST 데이터셋은 [Yann LeCun의 웹사이트](http://yann.lecun.com/exdb/mnist/)에 호스팅되어 있습니다. 좀 편하게 하기 위해서, 우리는 이 데이터를 다운로드받고 설치하는 파이썬 코드를 넣어놨습니다. [여기](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/input_data.py)에서 코드를 다운받은 후 아래와 같이 코드를 불러올 수도 있습니다. 아니면 그냥 복사하고 붙여넣기를 하십시오.

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

다운로드된 데이터는 55,000개의 학습 데이터(`mnist.train`), 10,000개의 테스트 데이터(`mnist.text`), 그리고 5,000개의 검증 데이터(`mnist.validation`) 이렇게 세 부분으로 나뉩니다. 데이터가 이렇게 나뉜다는 것은 매우 중요합니다. 왜냐하면 우리가 학습시키지 않는 데이터를 통해, 우리가 학습한 것이 정말로 일반화되었다고 확신할 수 있기 때문입니다!

앞서 언급했듯이, 각 MNIST 데이터셋은 두 부분으로 나뉩니다. 손으로 쓴 숫자와 그에 따른 라벨입니다. 우리는 이미지를 "xs"라고 부르고 라벨을 "ys"라고 부를 것입니다. 학습 데이터셋과 테스트 데이터셋은 둘 다 xs와 ys를 가집니다. 예를 들어, 학습 이미지는 `mnist.train.images`이며, 학습 라벨은 `mnist.train.labels`입니다.

각 이미지는 28x28 픽셀입니다. 우리는 이를 숫자의 큰 배열로 해석할 수 있습니다.

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/MNIST-Matrix.png">
</div>

우리는 이 배열을 펼쳐서 28x28 = 784 개의 벡터로 만들 수 있습니다. 이미지들 간에 일관적으로 처리하기만 한다면, 배열을 어떻게 펼치든지 상관없습니다. 이러한 관점에서, MNIST 이미지는 [매우 호화스러운 구조](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)(주의 : 연산을 많이 요하는 시각화입니다)를 가진, 단지 784차원 벡터 공간에 있는 여러 개의 데이터일 뿐입니다.

데이터를 펼치면 이미지의 2D 구조에 대한 정보가 완벽하게 사라집니다. 그게 나빠보이지 않나요? 음, 가장 좋은 컴퓨터 비전의 방법론은 2D 구조를 쓰고, 나중의 튜토리얼에서 다룹니다. 하지만 여기서 사용하는 간단한 소프트맥스 회귀는 그렇지 않습니다.

데이터를 펼친 결과로 `mnist.train.images`는 `[55000, 784]`의 형태를 가진 텐서(n차원 배열)가 됩니다. 첫 번째 차원은 이미지를 가리키며, 두 번째 차원은 각 이미지의 픽셀을 가르킵니다. 텐서의 모든 성분은 특정 이미지의 특정 픽셀을 특정하는 0과 1사이의 픽셀 강도입니다.

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/mnist-train-xs.png">
</div>

MNIST에서 각각에 대응하는 라벨은 0과 9사이의 숫자이며, 각 이미지가 어떤 숫자인지를 말해줍니다. 이 튜토리얼의 목적을 위해서 우리는 라벨을 "원-핫 벡터"로 바꾸길 원합니다. 원-핫 벡터는 단 하나의 차원에서만 1이고, 나머지 차원에서는 0인 벡터입니다. 이 경우, n번째 숫자는 n번째 차원이 1인 벡터로 표현될 것입니다. 예를 들어서, 3은 `[0,0,0,1,0,0,0,0,0,0]`입니다. 결과적으로, `mnist.train.labels`는 `[55000, 10]`의 모양을 같은 실수 배열이 됩니다.(*역자 주 : 정수 배열이 아니라, 실수 배열로 취급하는 데에는 이후 소프트맥스 회귀의 결과가 정수형이 아닌 실수형으로 산출되기 때문입니다.*)

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/mnist-train-ys.png">
</div>

그럼 이제 실제로 우리의 모델을 만들 준비가 되었습니다.

## Softmax Regressions

We know that every image in MNIST is a digit, whether it's a zero or a nine. We
want to be able to look at an image and give probabilities for it being each
digit. For example, our model might look at a picture of a nine and be 80% sure
it's a nine, but give a 5% chance to it being an eight (because of the top loop)
and a bit of probability to all the others because it isn't sure.

This is a classic case where a softmax regression is a natural, simple model.
If you want to assign probabilities to an object being one of several different
things, softmax is the thing to do. Even later on, when we train more
sophisticated models, the final step will be a layer of softmax.

A softmax regression has two steps: first we add up the evidence of our input
being in certain classes, and then we convert that evidence into probabilities.

To tally up the evidence that a given image is in a particular class, we do a
weighted sum of the pixel intensities. The weight is negative if that pixel
having a high intensity is evidence against the image being in that class,
and positive if it is evidence in favor.

The following diagram shows the weights one model learned for each of these
classes. Red represents negative weights, while blue represents positive
weights.

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/softmax-weights.png">
</div>

We also add some extra evidence called a bias. Basically, we want to be able
to say that some things are more likely independent of the input. The result is
that the evidence for a class \\(i\\) given an input \\(x\\) is:

$$\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i$$

where \\(W_i\\) is the weights and \\(b_i\\) is the bias for class \\(i\\),
and \\(j\\) is an index for summing over the pixels in our input image \\(x\\).
We then convert the evidence tallies into our predicted probabilities
\\(y\\) using the "softmax" function:

$$y = \text{softmax}(\text{evidence})$$

Here softmax is serving as an "activation" or "link" function, shaping
the output of our linear function into the form we want -- in this case, a
probability distribution over 10 cases.
You can think of it as converting tallies
of evidence into probabilities of our input being in each class.
It's defined as:

$$\text{softmax}(x) = \text{normalize}(\exp(x))$$

If you expand that equation out, you get:

$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

But it's often more helpful to think of softmax the first way:
exponentiating its inputs and then normalizing them.
The exponentiation means that one more unit of evidence increases the weight
given to any hypothesis multiplicatively.
And conversely, having one less unit of evidence means that a
hypothesis gets a fraction of its earlier weight. No hypothesis ever has zero
or negative weight. Softmax then normalizes these weights, so that they add up
to one, forming a valid probability distribution. (To get more intuition about
the softmax function, check out the
[section](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)
on it in Michael Nielsen's book, complete with an interactive visualization.)


You can picture our softmax regression as looking something like the following,
although with a lot more \\(x\\)s. For each output, we compute a weighted sum of
the \\(x\\)s, add a bias, and then apply softmax.

<div style="width:55%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/softmax-regression-scalargraph.png">
</div>

If we write that out as equations, we get:

<div style="width:52%; margin-left:25%; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/softmax-regression-scalarequation.png">
</div>

We can "vectorize" this procedure, turning it into a matrix multiplication
and vector addition. This is helpful for computational efficiency. (It's also
a useful way to think.)

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../../images/softmax-regression-vectorequation.png">
</div>

More compactly, we can just write:

$$y = \text{softmax}(Wx + b)$$


## Implementing the Regression


To do efficient numerical computing in Python, we typically use libraries like
NumPy that do expensive operations such as matrix multiplication outside Python,
using highly efficient code implemented in another language.
Unfortunately, there can still be a lot of overhead from switching back to
Python every operation. This overhead is especially bad if you want to run
computations on GPUs or in a distributed manner, where there can be a high cost
to transferring data.

TensorFlow also does its heavy lifting outside python,
but it takes things a step further to avoid this overhead.
Instead of running a single expensive operation independently
from Python, TensorFlow lets us describe a graph of interacting operations that
run entirely outside Python. (Approaches like this can be seen in a few
machine learning libraries.)

To use TensorFlow, we need to import it.

```python
import tensorflow as tf
```

We describe these interacting operations by manipulating symbolic variables.
Let's create one:

```python
x = tf.placeholder(tf.float32, [None, 784])
```

`x` isn't a specific value. It's a `placeholder`, a value that we'll input when
we ask TensorFlow to run a computation. We want to be able to input any number
of MNIST images, each flattened into a 784-dimensional vector. We represent
this as a 2-D tensor of floating-point numbers, with a shape `[None, 784]`.
(Here `None` means that a dimension can be of any length.)

We also need the weights and biases for our model. We could imagine treating
these like additional inputs, but TensorFlow has an even better way to handle
it: `Variable`.
A `Variable` is a modifiable tensor that lives in TensorFlow's graph of
interacting
operations. It can be used and even modified by the computation. For machine
learning applications, one generally has the model parameters be `Variable`s.

```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

We create these `Variable`s by giving `tf.Variable` the initial value of the
`Variable`: in this case, we initialize both `W` and `b` as tensors full of
zeros. Since we are going to learn `W` and `b`, it doesn't matter very much
what they initially are.

Notice that `W` has a shape of [784, 10] because we want to multiply the
784-dimensional image vectors by it to produce 10-dimensional vectors of
evidence for the difference classes. `b` has a shape of [10] so we can add it
to the output.

We can now implement our model. It only takes one line!

```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

First, we multiply `x` by `W` with the expression `tf.matmul(x, W)`. This is
flipped from when we multiplied them in our equation, where we had \\(Wx\\), as a
small trick
to deal with `x` being a 2D tensor with multiple inputs. We then add `b`, and
finally apply `tf.nn.softmax`.

That's it. It only took us one line to define our model, after a couple short
lines of setup. That isn't because TensorFlow is designed to make a softmax
regression particularly easy: it's just a very flexible way to describe many
kinds of numerical computations, from machine learning models to physics
simulations. And once defined, our model can be run on different devices:
your computer's CPU, GPUs, and even phones!


## Training

In order to train our model, we need to define what it means for the  model to
be good. Well, actually, in machine learning we typically define what it means
for a model to be bad, called the cost or loss, and then try to minimize how bad
it is. But the two are equivalent.

One very common, very nice cost function is "cross-entropy." Surprisingly,
cross-entropy arises from thinking about information compressing codes in
information theory but it winds up being an important idea in lots of areas,
from gambling to machine learning. It's defined:

$$H_{y'}(y) = -\sum_i y'_i \log(y_i)$$

Where \\(y\\) is our predicted probability distribution, and \\(y'\\) is the true
distribution (the one-hot vector we'll input).  In some rough sense, the
cross-entropy is measuring how inefficient our predictions are for describing
the truth. Going into more detail about cross-entropy is beyond the scope of
this tutorial, but it's well worth
[understanding](http://colah.github.io/posts/2015-09-Visual-Information/).

To implement cross-entropy we need to first add a new placeholder to input
the correct answers:

```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

Then we can implement the cross-entropy, \\(-\sum y'\log(y)\\):

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

First, `tf.log` computes the logarithm of each element of `y`. Next, we multiply
each element of `y_` with the corresponding element of `tf.log(y)`. Then 
`tf.reduce_sum` adds the elements in the second dimension of y, due to the 
`reduction_indices=[1]` parameter. Finally,  `tf.reduce_mean` computes the mean
over all the examples in the batch.

Now that we know what we want our model to do, it's very easy to have TensorFlow
train it to do so.
Because TensorFlow knows the entire graph of your computations, it
can automatically use the [backpropagation
algorithm](http://colah.github.io/posts/2015-08-Backprop/)
to efficiently determine how your variables affect the cost you ask it to
minimize. Then it can apply your choice of optimization algorithm to modify the
variables and reduce the cost.

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

In this case, we ask TensorFlow to minimize `cross_entropy` using the gradient
descent algorithm with a learning rate of 0.5. Gradient descent is a simple
procedure, where TensorFlow simply shifts each variable a little bit in the
direction that reduces the cost. But TensorFlow also provides
[many other optimization algorithms]
(../../../api_docs/python/train.md#optimizers): using one is as simple as
tweaking one line.

What TensorFlow actually does here, behind the scenes, is it adds new operations
to your graph which
implement backpropagation and gradient descent. Then it gives you back a
single operation which, when run, will do a step of gradient descent training,
slightly tweaking your variables to reduce the cost.

Now we have our model set up to train. One last thing before we launch it,
we have to add an operation to initialize the variables we created:

```python
init = tf.initialize_all_variables()
```

We can now launch the model in a `Session`, and run the operation that
initializes the variables:

```python
sess = tf.Session()
sess.run(init)
```

Let's train -- we'll run the training step 1000 times!

```python
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

Each step of the loop, we get a "batch" of one hundred random data points from
our training set. We run `train_step` feeding in the batches data to replace
the `placeholder`s.

Using small batches of random data is called stochastic training -- in
this case, stochastic gradient descent. Ideally, we'd like to use all our data
for every step of training because that would give us a better sense of what
we should be doing, but that's expensive. So, instead, we use a different subset
every time. Doing this is cheap and has much of the same benefit.



## Evaluating Our Model

How well does our model do?

Well, first let's figure out where we predicted the correct label. `tf.argmax`
is an extremely useful function which gives you the index of the highest entry
in a tensor along some axis. For example, `tf.argmax(y,1)` is the label our
model thinks is most likely for each input, while `tf.argmax(y_,1)` is the
correct label. We can use `tf.equal` to check if our prediction matches the
truth.

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

That gives us a list of booleans. To determine what fraction are correct, we
cast to floating point numbers and then take the mean. For example,
`[True, False, True, True]` would become `[1,0,1,1]` which would become `0.75`.

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Finally, we ask for our accuracy on our test data.

```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

This should be about 92%.

Is that good? Well, not really. In fact, it's pretty bad. This is because we're
using a very simple model. With some small changes, we can get to
97%. The best models can get to over 99.7% accuracy! (For more information, have
a look at this
[list of results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).)

What matters is that we learned from this model. Still, if you're feeling a bit
down about these results, check out [the next tutorial](../../../tutorials/mnist/pros/index.md) where we
do a lot better, and learn how to build more sophisticated models using
TensorFlow!
