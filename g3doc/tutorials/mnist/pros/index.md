#  MNIST 고급

TensorFlow는 큰 규모의 수치 계산에 적합한 강력한 라이브러리입니다. TensorFlow가 강력한 힘을 발휘하는 작업 중 하나는, 심층 신경망을 구성하고 학습시키는 것입니다. 이 튜토리얼에서는 MNIST 데이터를 분류하는 심층 합성곱(convolutional) 신경망을 구성하면서, TensorFlow에서 신경망 모델을 구성하는 기본 블록에 대해 알아볼 것입니다.

*이 튜토리얼은 인공 신경망과 MNIST 데이터셋에 익숙한 독자를 위해 구성되어 있습니다. 만약 이들에 익숙하지 않다면, [MNIST 초급](../beginners/index.md) 튜토리얼이 도움이 될 것입니다. 진행하기 전, [Tensorflow가 설치](../../../get_started/os_setup.md)되어 있는지 확인해 주세요. *

## 설정

모델을 생성하기 전, 먼저 MNIST 데이터셋을 불러오고, TensorFlow 세션을 시작할 것입니다.

### MNIST 데이터셋 불러오기

편의를 위해서, 자동으로 MNIST 데이터셋을 다운받은 뒤 불러오는 [스크립트](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/input_data.py)가 준비되어 있습니다. 아래와 같이 해당 스크립트를 import 하여 실행하면, 현재 디렉토리 하위에 `'MNIST_data'` 폴더를 생성하여 자동으로 데이터 파일을 저장할 것입니다.

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```
위에서 `mnist`는 훈련(training), 테스트(testing) 그리고 검증(validation) 데이터를 NumPy 배열로 저장하는 클래스입니다. 아래에서 사용될 미니배치(minibatch)를 추출하는 함수 또한 이 클래스 안에 포함되어 있습니다.

### TensorFlow InteractiveSession 시작하기

TensorFlow는 계산을 위해 고효율의 C++ 백엔드(backend)를 사용합니다. 이 백엔드와의 연결을 위해 TensorFlow는 세션(session)을 사용합니다. 일반적으로 TensorFlow 프로그램은 먼저 그래프를 구성하고, 그 이후 그래프를 세션을 통해 실행하는 방식을 따릅니다.

여기서는 대신 TensorFlow 코드를 보다 유연하게 작성할 수 있게 해 주는 `InteractiveSession` 클래스를 사용할 것입니다. 이 클래스는 [계산 그래프](../../../get_started/basic_usage.md#the-computation-graph)(computation graph)를 구성하는 작업과 그 그래프를 실행하는 작업을 분리시켜 줍니다. 즉, `InteractiveSession`을 쓰지 않는다면, 세션을 시작하여 [그래프를 실행](../../../get_started/basic_usage.md#launching-the-graph-in-a-session)하기 전에 이미 전체 계산 그래프가 구성되어 있어야 하는 것입니다.

```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

#### 계산 그래프 (Computational Graph)

Python에서 효율적인 수치 계산을 하기 위해서, 주로 NumPy와 같이 Python 외부에서 다른 언어로 된 고효율의 코드를 통해 행렬 곱셈과 같은 고비용의(expensive) 연산을 수행하는 라이브러리를 이용합니다. 불행히도, 이렇게 하면 연산 결과를 일일이 Python으로 다시 불러들이는 데 많은 오버헤드가 발생합니다. 특히 계산 과정을 여러 GPU에 분산시키는 경우, 데이터를 이동시키는 데 드는 비용이 매우 커지게 됩니다.

TensorFlow도 마찬가지로 고비용의 연산은 Python 외부에서 실행합니다. 하지만, 위와 같은 오버헤드 문제를 피하기 위해 현명한 방법을 활용합니다. 각각의 고비용 연산을 Python에서 독립적으로 실행하는 대신, TensorFlow는 상호작용하는 연산을 그래프로 묶어 그 전체를 Python 바깥에서 실행시키는 방법을 사용합니다. Theano나 Torch와 같은 라이브러리에서 활용되는 방법과 비슷합니다.

따라서 Python에서 작성하는 코드의 역할은, 이러한 외부의 계산 그래프를 구성하고, 이 계산 그래프의 어떤 부분이 실행되어야 하는지 지시하는 것입니다. 자세한 내용은 [계산 그래프](../../../get_started/basic_usage.md#the-computation-graph) 및 [기본 사용법](../../../get_started/basic_usage.md)을 참고하세요.

## 소프트맥스 회귀 모델 구성

이 절에서는 단일 계층의 소프트맥스 회귀 모델(softmax regression model)을 구성할 것입니다. 그리고 다음 절에서 이를 확장시켜, 다중 계층의 합성곱 신경망(convolutional network)을 구성할 것입니다.

### 플레이스홀더 (Placeholder)

계산 그래프를 구성하기 위해, 먼저 입력될 이미지와 각각의 출력 클래스에 해당하는 노드를 생성할 것입니다.

```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

위 코드에서 `x` 와 `y_` 에 특정한 값이 부여된 것은 아닙니다. 그들은 나중에 TensorFlow가 계산을 실행할 때 값을 넣어 줄 자리인 `placeholder` 입니다.

입력될 이미지들 `x`는 부동 소수점 실수(float) 값들의 2D 텐서입니다. 위 코드에서 `shape`에 `[None, 784]`를 넣어 주었는데, 여기서 `784`는 28x28의 크기를 가지는 MNIST 이미지를 한 줄로 펼친 크기에 해당합니다. 배치(batch)의 크기에 해당하는 첫 번째 차원 크기의 `None`은 크기를 여기서 정하지 않는다(어떤 배치 크기라도 가능하다)는 것을 의미합니다. 출력 클래스인 `y_` 또한 2D 텐서입니다. 각 열은 해당하는 MNIST 이미지의 숫자 클래스를 10차원 one-hot 벡터로 나타냅니다.

`tf.placeholder`에 `shape` 매개변수가 필수는 아닙니다. 하지만, 이를 명시해 줌으로써 TensorFlow가 잘못된 텐서 구조(shape)에 따른 오류를 자동으로 잡아낼 수 있게 됩니다.

### 변수 (Variable)

이제 모델에 사용할 가중치(weight) `W`와 편향(bias) `b`를 정의합니다. 이들을 추가적인 입력으로 대할 수도 있겠지만, TensorFlow는 이러한 변수들을 다루기 위해 `Variable`을 제공합니다. `Variable`이란 TensorFlow의 계산 그래프 안에 있는 값입니다. 이들은 계산에 사용될 수 있을 뿐만 아니라, 계산에 의해 변경될 수도 있습니다. 따라서 머신 러닝에 활용되는 모델 매개변수는 주로 `Variable`들로 구성됩니다.

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

`tf.Variable`을 사용할 때에는 변수의 초기 값을 지정해 주어야 합니다. 위의 경우, `W`와 `b` 모두 0으로만 구성된 텐서로 초기화됩니다.`W`는 784x10 행렬(입력 이미지 벡터의 크기가 784, 출력 숫자 클래스가 10개)이며, `b`는 10차원 벡터입니다.

`Variable`들은 세션이 시작되기 전에 초기화되어야 합니다. 아래 코드는 모든 `Variable`들 각각에 대해 미리 지정된 초기 값(위에서 지정된 0으로만 구성된 텐서)를 넣어 주는 역할을 합니다.

```python
sess.run(tf.global_variables_initializer())
```

### 클래스 예측 및 비용 함수(Cost Function)

이제 회귀 모델을 도입할 수 있습니다. 한 줄만으로요! 벡터화된 입력 이미지인 `x`를 가중치 행렬인 `W`와 곱하고, 여기에 편향 `b`를 더한 뒤, 각각의 클래스에 대한 소프트맥스 함수의 결과를 계산하면 됩니다.

```python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```

모델 훈련 과정에서 최소화될 비용 함수(cost function) 또한 간단하게 도입할 수 있습니다. 여기서 사용될 비용 함수는 실제 클래스와 모델의 예측 결과 간 크로스 엔트로피(cross-entropy) 함수입니다.

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

여기서 `tf.reduce_sum`은 모든 클래스에 대해 결과를 합하는 함수, `tf.reduce_mean`은 사용된 이미지들 각각에서 계산된 합의 평균을 구하는 함수입니다.

## 모델 훈련시키기

이제 모델과 훈련의 비용 함수가 정의되었으니, TensorFlow로 모델을 훈련시키는 일만 남았습니다. TensorFlow에 전체 계산 그래프의 정보가 입력되어 있으므로, 라이브러리가 자동으로 미분을 통해 각각의 변수에 대한 비용 함수의 기울기(gradient)를 계산합니다. TensorFlow는 다양한 [내장된 최적화 알고리즘](../../../api_docs/python/train.md#optimizers)을 가지고 있습니다. 여기서는 아래 코드와 같이 학습 속도 0.5의 경사 하강법(steepest gradient descent) 알고리즘을 사용하여 크로스 엔트로피를 최소화할 것입니다.

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

위의 코드 한 줄에서 TensorFlow가 실제로 하는 것은 계산 그래프에 기울기를 계산하고, 얼마나 매개변수를 변경해야 할지 계산하고, 매개변수를 변경하는 새로운 계산들을 추가하는 것입니다.

반환된 `train_step`은 실행되었을 때 경사 하강법을 통해 각각의 매개변수를 변화시키게 됩니다. 따라서, 모델을 훈련시키려면 이 `train_step`을 반복해서 실행하면 됩니다.

```python
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

각각의 훈련 단계(iteration)에서, 50개의 훈련 샘플이 추출됩니다. 그리고 `train_step`을 실행하며 `feed_dict`를 통해 `placeholder` 텐서인 `x`와 `y_`에 훈련 샘플을 넣어줍니다. 참고로, `feed_dict`는 `placeholder` 외에도 계산 그래프 안의 어떤 텐서라도 변경할 수 있습니다.

### 모델 평가하기

이렇게 훈련된 모델은 얼마나 정확할까요?

먼저, 모델이 정확한 레이블을 예측했는지 확인해 볼 것입니다. `tf.argmax` 함수는 텐서의 한 차원을 따라 가장 큰 값의 인덱스를 반환합니다. 예로, `tf.argmax(y,1)`은 모델이 입력을 받고 가장 그럴듯하다고 생각한 레이블이고, `tf.argmax(y_,1)`은 실제 레이블입니다. 이제 `tf.equal` 함수를 사용해 두 레이블이 일치하는지 다음과 같이 확인할 수 있습니다.

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

위의 코드는 불리언으로 이루어진 리스트를 반환합니다. 전체에서 얼마나 맞았는지를 확인하려면, 불리언을 부동 소수점 실수로 형변환하여 리스트의 평균을 구하면 됩니다. 예로, 결과가 `[True, False, True, True]` 였다면 이는 형변환을 통해 `[1,0,1,1]` 이 되고, 평균인 `0.75`가 예측 결과의 정확도가 됩니다.

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

이제 아래와 같이 `feed_dict`로 `mnist.test`를 전달하여 테스트 데이터셋에 대한 예측 정확도를 확인할 수 있습니다. 대략 92% 정도의 정확도가 얻어질 것입니다.

```python
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

## 다중 계층 합성곱 신경망

MNIST 데이터에서 91% 정확도를 얻는 것은 그다지 좋은 결과라고 할 수 없습니다. 그래서 이번 장에서는, 정확도를 높이기 위해 합성곱 신경망(convolutional neural network)이라는 약간 복잡한 모형을 사용할 것입니다. 이를 통해 99.2% 정도의 정확도를 얻을 수 있습니다. 최신 결과에는 미치지 못하지만, 어느 정도 그럴듯한 결과입니다.

### 가중치 초기화

합성곱 신경망 모델을 구성하기 위해서는 많은 수의 가중치와 편향을 사용하게 됩니다. 대칭성을 깨뜨리고 기울기(gradient)가 0이 되는 것을 방지하기 위해, 가중치에 약간의 잡음을 주어 초기화합니다. 또한, 모델에 ReLU 뉴런이 포함되므로, "죽은 뉴런"을 방지하기 위해 편향을 작은 양수(0.1)로 초기화합니다. 매번 모델을 만들 때마다 반복하는 대신, 아래 코드와 같이 이러한 일을 해 주는 함수 두 개를 생성합니다.

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

### 합성곱(Convolution)과 풀링(Pooling)

TensorFlow는 합성곱과 풀링 계층(layer)을 유연하게 다룰 수 있도록 해 줍니다. 경계의 패딩(padding)과 스트라이드(stride)에 대해 다양한 선택을 할 수 있습니다. 이번 예시에서는 스트라이드를 1로, 출력 크기가 입력과 같게 되도록 0으로 패딩하도록 설정합니다. 풀링은 2x2 크기의 맥스 풀링을 적용합니다. 마찬가지로 코드를 간단히 하기 위해 합성곱과 풀링을 위한 함수를 아래 코드와 같이 생성합니다.

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

### 첫 번째 합성곱 계층

이제 첫 번째 계층을 만들 것입니다. 이는 합성곱 계층과 맥스 풀링 계층으로 구성됩니다. 합성곱 계층에서는 5x5의 윈도우(patch라고도 함) 크기를 가지는 32개의 필터를 사용하며, 따라서 구조(shape)가 `[5, 5, 1, 32]`인 가중치 텐서를 정의해야 합니다. 처음 두 개의 차원은 윈도우의 크기, 세 번째는 입력 채널의 수, 마지막은 출력 채널의 수(즉, 얼마나 많은 특징을 사용할 것인가)를 나타냅니다. 또한, 각각의 출력 채널에 대한 편향을 정의해야 합니다. 이 과정에서 앞에서 만든 함수를 사용합니다.

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

이 계층에 이미지를 입력하려면 먼저 `x`를 4D 텐서로 `reshape`해야 합니다. 두 번째와 세 번째 차원은 이미지의 가로와 세로 길이, 그리고 마지막 차원은 컬러 채널의 수를 나타냅니다.

```python
x_image = tf.reshape(x, [-1,28,28,1])
```

이제 `x_image`와 가중치 텐서에 합성곱을 적용하고, 편향을 더한 뒤 ReLU 함수를 적용합니다. 출력 값을 구하기 위해 마지막으로 맥스 풀링을 적용합니다.

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### 두 번째 합성곱 계층

심층 신경망을 구성하기 위해서, 앞에서 만든 것과 비슷한 계층을 쌓아올릴 수 있습니다. 여기서는 두 번째 합성곱 계층이 5x5 윈도우에 64개의 필터를 가집니다.

```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

### 완전 연결 계층 (Fully-Connected Layer)

두 번째 계층을 거친 뒤 이미지 크기는 7x7로 줄어들었습니다. 이제 여기에 1024개의 뉴런으로 연결되는 완전 연결 계층을 구성합니다. 이를 위해서 7x7 이미지의 배열을 `reshape`해야 하며, 완전 연결 계층에 맞는 가중치 행렬과 편향 행렬을 구성합니다. 최종적으로 완전 연결 계층의 끝에 ReLU 함수를 적용합니다.

```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

#### 드롭아웃 (Dropout)

오버피팅(overfitting) 되는 것을 방지하기 위해, [드롭아웃](
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)을 적용할 것입니다. 뉴런이 드롭아웃되지 않을 확률을 저장하는 `placeholder`를 만듭니다. 이렇게 하면 나중에 드롭아웃이 훈련 과정에는 적용되고, 테스트 과정에서는 적용되지 않도록 설정할 수 있습니다. TensorFlow의 `tf.nn.dropout` 함수는 뉴런의 출력을 자동으로 스케일링(scaling)하므로, 추가로 스케일링 할 필요 없이 그냥 드롭아웃을 적용할 수 있습니다.<sup id="a1">[1](#f1)</sup>

```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

### 최종 소프트맥스 계층

마지막으로, 위에서 단일 계층 소프트맥스 회귀 모델을 구성할 때와 비슷하게 아래 코드와 같이 소프트맥스 계층을 추가합니다.

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

### 모델의 훈련 및 평가

이렇게 훈련된 모델은 얼마나 정확할까요?

훈련 및 평가 또한 위의 단일 계층 모델과 거의 같습니다. 차이가 있다면, 이번에는 경사 하강법 알고리즘 대신 더 복잡한 ADAM 최적화 알고리즘을 사용합니다. 또한, 드롭아웃 확률을 설정하는 추가 변수인 `keep_prob`을 `feed_dict` 인수를 통해 전달합니다. 아래의 코드는 훈련 과정에서 100회 반복 시마다 로그를 작성합니다.

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

코드를 실행시켜서 얻은 최종 정확도는 약 99.2%가 될 것입니다.

이렇게 하여 TensorFlow를 이용해 쉽고 빠르게 '어느 정도 복잡한 딥 러닝 모델'을 구성하고, 훈련시키고, 평가하는 과정을 배워 보았습니다.

<b id="f1">1</b>: 드롭아웃은 오버피팅을 줄이는 데 매우 효과적이지만, 이번에 다룬 작은 합성곱 신경망에 대해서는, 성능이 드롭아웃을 적용한 경우와 하지 않은 경우가 비슷합니다. 드롭아웃은 큰 신경망을 훈련시킬 때 유용합니다. [↩](#a1)
