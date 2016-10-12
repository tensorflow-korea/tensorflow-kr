# Basic Usage

TensorFlow 를 사용하기 위해서는 TensorFlow 가 어떻게 동작하는지 이해할 필요가 있다.

* `Graphs`는 컴퓨터 연산을 나타냅니다.
* `Sessions`이라는 컨텍스트 안에서 그래프는 실행 됩니다.
* `Tensors`는 데이터를 나타냅니다.
* `Variables`은 상태를 유지합니다.
* `Feeds & Fetches`은 임의의 작업의 데이터 입출력에 사용 됩니다.

## Overview

TensorFlow 의 프로그램 시스템은 `Graphs`를 통해서 컴퓨터 연산을 나타낸다.  `Graph` 안에서 노드는
*ops* (작은 단위의 동작) 불리우며, 한개의 OP는 제로 또는 하나 이상의 `Tensors` 가지게 된다.
OP는 몇 가지의 연산을 하기 되고, 제로 또는 하나 이상의 `Tensors` 만들어 냅니다.
`Tensor` 는 다차원 배열을 타입으로 가지게 된다.
예를 들어, 이미지를 다음과 같이 `[batch, height, width, channels]`4차원 배열로 나타낼 수 있다.

TensorFlow의 graph는 컴퓨터 연산의 *description* 설명이다. 어떤것을 연산하든, graph는
반드시 `Session` 안에서 실행되어야 한다. `Session`은 graph 작은 단위 연산들을 CPUs 또는 GPUs
불리우는 `Devices`에 배치시키고, 작은 단위 연산(ops)들이 작동할수 있는 기능을 제공한다.
이런 기능은 단위 연산(ops)에 의해서 tensors 들이 만들어진다. 파이썬에서는 `ndarray` 오프젝트로
C 와 C++ 에서는 `TensorFlow::Tensor` 오프젝트를 만들어내게 된다.

## The computation graph

TensorFlow 프로그램들은 대개 graph를 조립하는 '구성 단계'와 session을 이용해 graph 안에
작은 단위의 연산(ops)을 실행시키는 '실행 단계'로 구성돼 있다.

`TensorFlow` 프로그램은 일반적으로 construction phase 안에서 구성된다. construction phase란
하나의 그래프에서 조립되어진다. 실행 phases 는 그래프에 있는 ops 를 실행시키기 위해서 `Session` 을
사용하게 된다.

예를 들어, 그래프는 construction phase 안에서 뉴럴 네트워크를 훈련시키고 나타내기 위해서
만들어 진다. 그리고 실행 phase 안의 그래프는 훈련셋(ops)를 반복적으로 실행하게 된다.
`TensorFlow` 는 C, C++, Pythons 프로그램에서 사용할수 있다. 현재 Python 라이버리를 사용하면
그래프를 쉽게 조립할 있다. 또한 C, C++에는 제공하지 않는 많은 헬퍼 함수들을 제공하고 있다.

`Sessions` 라이브러리는 3개의 언어를 위해서 환경 함수들을 가지고 있다.

### Building the graph
graph를 만드는 것은 `Constant`와 같이 어떠한 input도 필요하지 않는 단위의 동작(ops)으로 시작한다.
Python 라이브러리에서 단위 연산(ops) 생성자는 구성된 단위 연산(ops)의 결과(output)를 대기하는
객체를 반환한다. 그리고 이 객체들은 다른 단위 연산(ops) 생성자의 input으로 전달할 수 있다.

Python 라이브러리로 사용하는 TensorFlow는 단위 연산(ops) 생성자가 노드를 추가한
*graph* 를 가지고 된다. *graph* 는 많은 어플리케이션용으로 충분하다.
[Graph class](../api_docs/python/framework.md#Graph) 문서에서 어떻게 많은 graph를
명시적으로 관리할 수 있는지 알 수 있다.

```python
import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
```

예시 graph는 3개의 노드(`constant()` ops 2개와 `matmul()` ops 한개)를 가지고 있다.
실제 매트릭스들을 곱하고 곱셈한 연산의 결과를 얻기 위해선, session에서 graph를 실행해야 한다.

### Launching the graph in a session

아래와 같은 구성으로 동작하게 된다. 'graph' 동작 시키기 위해서는 `Session` 을 만든다.
예시 graph 에서는 변수 없이 session 이 동작하게 된다.

모든 session API 는 [Session class](../api_docs/python/client.md#session-management)
에서 볼수 있다.

```python
# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()
```

Sessions 은 자원을 해제하기 위해서 닫아야 한다. 또한 `Session`은 "with" 블럭과 함께
사용할 수 있다. `Session`은 `with` 블럭의 끝에서 자동으로 닫히게 된다. 

```python
with tf.Session() as sess:
  result = sess.run([product])
  print(result)
```

TensorFlow는 graph 에 정의된 단위 연산들이 컴퓨터 자원(CPU or GPU)을 분배해서 사용 할수 있게
구현되어 있다.
만약에 CPU or GPU 를 명시적으로 지정하지 않는다면, TensorFlow 는 당신이 가지고 있는 첫번째 GPU를
사용하게 될것이다. 이것은 보다 많은 연산처리를 가능하게 한다.

만약에 당신의 컴퓨터에 하나 이상의 GPU가 있다면, 당신은 명시적으로 GPU를 지정해서 사용할수 있다.
`with...Device` 라는 문법을 사용해서 연산에 사용될 CPU or GPU를 지정할 수 있다.

```python
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```

CPU or GPU 는 문자열로 지정 한다. 현재 제공하는 CPU or GPU는 아래와 같다.

*  `"/cpu:0"`: 컴퓨터의 CPU.
*  `"/gpu:0"`: 컴퓨터의 1번째 GPU.
*  `"/gpu:1"`: 컴퓨터의 2번쨰 GPU.

GPU 와 TensorFlow 보다 많은 정보는 [Using GPUs](../how_tos/using_gpu/index.md) 보면 된다.

### Launching the graph in a distributed session

TensorFlow 클러스터를 만들려면, 클러스트 안의 각 머신에서 TensorFlow 서버를 동작시켜야 한다.
세션을 클라이언트 안에서 인스턴스화 하려는 경우, 클러스트 안에 머신중 하나의 네트워크 경로를
전달해야 한다:

```python
with tf.Session("grpc://example.org:2222") as sess:
  # Calls to sess.run(...) will be executed on the cluster.
  ...
```
해당 머신은 현재 Session의 마스터가 된다. 마스터는 클러스터(workers) 안의 여러 머신들과 graph 를
교차해서 분배하게 된다. 이런 분배는 머신의 사용 가능한 컴퓨팅 자원을 고려해서 이루어 진다.

"with tf.device():" 문법을 이용해서 특정 머신에게 graph의 특정 연산을 지정해 줄 수 있다.

```python
with tf.device("/job:ps/task:0"):
  weights = tf.Variable(...)
  biases = tf.Variable(...)
```
Session 과 클러스터 의 분산처리에 대한 더 많은 정보는 [Distributed TensorFlow How To](../how_tos/distributed/)
볼 수 있다.

## Interactive Usage

이 문서에 있는 파이썬 예제들은 [`Session`](../api_docs/python/client.md#Session) 과
[`Session.run()`](../api_docs/python/client.md#Session.run) 함수를 사용해서 graph
의 연산들을 동작 시킨다.

인터렉티브 파이썬 환경 [IPython](http://ipython.org) 에서는 [`InteractiveSession`](../api_docs/python/client.md#InteractiveSession) 클래스,
[`Tensor.eval()`](../api_docs/python/framework.md#Tensor.eval) 와 [`Operation.run()`](../api_docs/python/framework.md#Operation.run) 함수를 사용 할 수 있다.
이것은 세션안에서 변수의 홀딩을 피할수 있기 한다.

```python
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()
```

## Tensors

TensorFlow 프로그램은 모든 데이터를 표현할수 있는 tensor 라는 데이터 구조를 사용한다.
tensor 만이 유일하게 연산 그래프 안의 연산자들 사이를 오가게 된다. TensorFlow 의 tensor 는
n-차원 배열 또는 리스트라고 생각하면 된다. Tensor 는 static type, rank, shape 을 가지고 있다.
TensorFlow 에서 이런 컨셉을 제대로 다루기 위해서 [Rank, Shape, and Type](../resources/dims_types.md) 을 참고 하자.

## Variables

Variables는 graph 에서 실행된 상대를 유지하게 된다. 아래의 심플한 카운터 예제를 통해서 variables
에 대해서 보여 준다. 더 자세한 내용은 [Variables](../how_tos/variables/index.md) 참조.

```python
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

# output:

# 0
# 1
# 2
# 3
```
코드안에서 `assign()` 연산은 graph 의 한부분으로 `add()` 처럼 동작하게 된다.
그리고 이런 연산은 `run()`이 실행되기 전까지 실제로 실행되지 않는다.

일반적으로 Variables 셋은 통계 모델의 파라메터를 나타낸다. 예를 들어, Variable 안의 tensor 에
뉴럴 네트워크를 위한 무게를 저장 한다면, 트레이닝 하는 동안 graph 는 반복적으로 tensor를 업데이트
하게 된다.

## Fetches

연산의 결과를 가져오는 것, `Session` 오브젝트 안에서 `run()` 호출은 graph 를 실행 시키고 tensors
결과를 끌어 내게 된다.
이전의 예제에서는 한개의 노드 `state` 만 가져왔다, 하지만 여러개의 tensors 도 가져 올 수 있다.

```python
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# output:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
```

All the ops needed to produce the values of the requested tensors are run once
(not once per requested tensor).

## Feeds

위의 예제에서 살펴본 graph에 tensor들은 `Constants` 와 `Variables` 에 저장되어 있다.
TensorFlow는 graph의 연산에게 직접 tensor의 값을 줄 수 있는 feed 메카니즘을 제공한다.

Feed 값에 따라 연산의 출력값이 대체 된다. feed 데이터의 변수는 `run()`이 제공된다. Feed 는 오직 `run()`
에서만 사용 되어 진다. 가장 일반적인 사용방법은 tf.placeholder() 을 사용해서 "feed" 작업을 지정해 주는것이다.

```python

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]
```

만약에 feed 를 제대로 제공하지 않는다면 `placeholder()` 연산은 에러를 만들게 된다.
feeds 다양한 사용 예를 보고 싶다면
[MNIST fully-connected feed tutorial](../tutorials/mnist/tf/index.md)
([source code](https://www.tensorflow.org/code/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py)) 참고
