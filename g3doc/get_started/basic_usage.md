# 기본 사용법(Basic Usage)
(v0.12rc1)

TensorFlow를 사용하기 위해 먼저 TensorFlow가 어떻게 동작하는지를 이해해 봅시다.

* 연산은 graph로 표현합니다.(역자 주: graph는 점과 선, 업계 용어로는 노드와 엣지로 이뤄진 수학적인 구조를 의미합니다.)
* graph는 `Session`내에서 실행됩니다.
* 데이터는 tensor로 표현합니다.
* `변수(Variable)`는 (역자 주: 여러 graph들이 작동할 때도) 그 상태를 유지합니다.
* 작업(operation 혹은 op)에서 데이터를 입출력 할 때 feed와 fetch를 사용할 수 있습니다.

## 개요(Overview)

TensorFlow는  graph로 연산(역자 주: 앞으로도 'computation'을 연산으로 번역합니다)을 나타내는 프로그래밍 시스템입니다. graph에 있는 노드는 *작업(op)*(작업(operation)의 약자)라고 부릅니다. 작업(op)은 0개 혹은 그 이상의 `Tensor`를 가질 수 있고 연산도 수행하며 0개 혹은 그 이상의 `Tensor`를 만들어 내기도 합니다. Tensorflow에서 `Tensor`는 정형화된 다차원 배열(a typed multi-dimensional array)입니다. 예를 들어, 이미지는 부동소수점 수(floating point number)를 이용한 4차원 배열(`[batch, height(가로), width(세로), channels(역자 주: 예를 들어 RGB)]`)로 나타낼 수 있습니다.

TensorFlow에서 graph는 연산을 *표현*해놓은 것이라서 연산을 하려면 graph가 `Session` 상에 실행되어야 합니다. `Session`은 graph의 작업(op)(역자 주: operation. graph를 구성하는 노드)을 CPU나 GPU같은 `Device`에 배정하고 실행을 위한 메서드들을 제공합니다. 이런 메서드들은 작업(op)을 실행해서 tensor를 만들어 냅니다. tensor는 파이썬에서 [numpy](http://www.numpy.org) `ndarray` 형식으로 나오고 C 와 C++ 에서는 `TensorFlow::Tensor` 형식으로 나옵니다.

## 연산 graph(The computation graph)

TensorFlow 프로그램은 보통 graph를 조립하는 '구성 단계(construction phase)'와 session을 이용해 graph의 op을 실행시키는 '실행 단계(execution phase)'로 구성됩니다.

예를 들어 뉴럴 네트워크를 표현하고 학습시키기 위해 구성 단계에는 graph를 만들고 실행 단계에는 graph의 훈련용 작업들(set of training ops)을 반복해서 실행합니다.

TensorFlow는 C, C++, 파이썬을 이용해서 쓸 수 있습니다. 지금은 graph를 만들기 위해 파이썬 라이브러리(역자 주: '파이썬 라이브러리'는 파이썬 라이브러리로 나온 TensorFlow를 의미합니다.)를 사용하는 것이 훨씬 쉽습니다. C, C++에서는 제공하지 않는 많은 헬퍼 함수들을 쓸 수 있기 때문이죠.

(역자 주: TensorFlow의) session 라이브러리들은 3개 언어에 동일한 기능을 제공합니다.

### graph 만들기(Building the graph)
graph를 만드는 것은 `상수(constant)`같이 아무 입력값이 필요없는 작업(op)을 정의하는 것에서부터 시작합니다. 이 op을 연산이 필요한 다른 op들에게 입력값으로 제공하는 것입니다.

파이썬 라이브러리의 작업 생성 함수(op constructor)는 만들어진 작업(op)들의 결과값을 반환합니다. 반환된 작업들의 결과값은 다른 작업(op)을 생성할 때 함수의 입력값으로 이용할 수 있습니다.

파이썬 라이브러리에는 작업 생성 함수로 노드를 추가할 수 있는 *default graph*라는 것이 있습니다. default graph는 다양하게 이용하기 좋습니다. [Graph class](../api_docs/python/framework.md#Graph) 문서에서 여러 그래프를 명시적으로 관리하는 방법을 알 수 있습니다.

```python
import tensorflow as tf

# 1x2 행렬을 만드는 constant op을 만들어 봅시다.
# 이 op는 default graph에 노드로 들어갈 것입니다.
# Create a constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# 생성함수에서 나온 값은 constant op의 결과값입니다.
# The value returned by the constructor represents the output
# of the constant op.
matrix1 = tf.constant([[3., 3.]])

# 1x2 행렬을 만드는 constant op을 만들어봅시다.
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# 'matrix1'과 'matrix2를 입력값으로 하는 Matmul op(역자 주: 행렬곱 op)을
# 만들어 봅시다.
# 이 op의 결과값인 'product'는 행렬곱의 결과를 의미합니다.
# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
```

default graph에는 이제 3개의 노드가 있습니다. 2개는 `상수(constant)` 작업(op)이고 하나는 `행렬곱(matmul)` 작업(op)이죠. 그런데 사실 행렬을 곱해서 결과값을 얻으려면 `Session`에다 graph를 *실행*해야 합니다.

### session에서 graph 실행하기(Launching the graph in a session)

graph를 구성하고 나면 `Session` 오브젝트를 만들어서 graph를 실행할 수 있습니다. op 생성함수에서 다른 graph를 지정해줄 때까지는 default graph가 `Session`에서 실행됩니다. 관련 내용은 [Session class](../api_docs/python/client.md#session-management)에서 확인할 수 있습니다.

```python
# default graph를 실행시켜 봅시다.
# Launch the default graph.
sess = tf.Session()

# 행렬곱 작업(op)을 실행하기 위해 session의 'run()' 메서드를 호출해서 행렬곱 
# 작업의 결과값인 'product' 값을 넘겨줍시다. 그 결과값을 원한다는 뜻입니다.
# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# 작업에 필요한 모든 입력값들은 자동적으로 session에서 실행되며 보통은 병렬로 
# 처리됩니다.
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# 'run(product)'가 호출되면 op 3개가 실행됩니다. 2개는 상수고 1개는 행렬곱이죠.
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# 작업의 결과물은 numpy `ndarray` 오브젝트인 result' 값으로 나옵니다.
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# 실행을 마치면 Session을 닫읍시다.
# Close the Session when we're done.
sess.close()
```

연산에 쓰인 시스템 자원을 돌려보내려면 session을 닫아야 합니다. 시스템 자원을 더 쉽게 관리하려면 `with` 구문을 쓰면 됩니다. 각 `Session`에 컨텍스트 매니저(역자 주: 파이썬의 요소 중 하나로 주로 'with' 구문에서 쓰임)가 있어서 'with' 구문 블락의 끝에서 자동으로 'close()'가 호출됩니다.

```python
with tf.Session() as sess:
  result = sess.run([product])
  print(result)
```

TensorFlow의 구현 코드(TensorFlow implementation)를 통해 graph에 정의된 내용이 실행가능한 작업들(operation)로 변환되고 CPU나 GPU같이 이용가능한 연산 자원들에 뿌려집니다. 코드로 어느 CPU 혹은 GPU를 사용할 지 명시적으로 지정할 필요는 없습니다. 작업을 가능한 한 많이 처리하기 위해 TensorFlow는 (컴퓨터가 GPU를 가지고 있다면) 첫 번째 GPU를 이용하니까요.

만약 컴퓨터에 복수의 GPU가 있어서 이를 사용하려면, op을 어느 하드웨어에 할당할 지 명시적으로 밝혀야 합니다. 작업에 사용할 CPU 혹은 GPU를 지정하려면 `with...Device` 구문을 사용하면 됩니다.

```python
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```

이용할 CPU 혹은 GPU는 문자열로 지정할 수 있습니다. 현재 지원되는 것은 아래와 같습니다.

*  `"/cpu:0"`: 컴퓨터의 CPU.
*  `"/gpu:0"`: 컴퓨터의 1번째 GPU.
*  `"/gpu:1"`: 컴퓨터의 2번쨰 GPU.

GPU와 TensorFlow에 대한 더 자세한 정보는 [Using GPUs](../how_tos/using_gpu/index.md)를 참조하시기 바랍니다.

### 분산된 session에서 graph 실행하기(Launching the graph in a distributed session)

TensorFlow 클러스터를 만들기 위해 클러스터에 포함된 각 머신에 TensorFlow 서버를 실행시켜 봅시다. Session을 클라이언트에 인스턴스화할 때는 Session을 클러스터 머신의 네트워크 위치로 넘겨야 합니다.:

```python
with tf.Session("grpc://example.org:2222") as sess:
  # sess.run(...)을 호출하면 클러스터에서 실행될 것입니다.
  # Calls to sess.run(...) will be executed on the cluster.
  ...
```

세션을 받은 머신은 해당 Session의 마스터가 됩니다. 머신 내에서는 Tensorflow의 구현 코드(implementation)가 머신 내 연산 자원에게 작업을 나눠주지만, 클러스터에서는 마스터가 클러스터 내의 다른 머신들에게 graph를 분배하는 것입니다.

"with tf.device():" 구문을 이용해서 특정 머신에게 직접 graph의 특정 부분을 지정해 줄 수도 있습니다.

```python
with tf.device("/job:ps/task:0"):
  weights = tf.Variable(...)
  biases = tf.Variable(...)
```

Session 분산과 클러스터에 대한 더 자세한 정보는 [Distributed TensorFlow How To](../how_tos/distributed/)를 참조하시기 바랍니다.

## 인터렉티브한 이용법(Interactive Usage)

이 문서에 있는 파이썬 예제들은 [`Session`](../api_docs/python/client.md#Session)을 실행시키고 [`Session.run()`](../api_docs/python/client.md#Session.run) 메서드를 이용해서 graph의 작업들을 처리한다.

[IPython](http://ipython.org)같은 인터렉티브 파이썬 환경에서의 이용편의성을 위해[`InteractiveSession`](../api_docs/python/client.md#InteractiveSession)클래스와 [`Tensor.eval()`](../api_docs/python/framework.md#Tensor.eval),[`Operation.run()`](../api_docs/python/framework.md#Operation.run) 메서드를 대신 이용할 수도 있다. session 내에서 변수를 계속 유지할 필요가 없기 때문이다.

```python
# 인터렉티브 TensorFlow Session을 시작해봅시다.
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 초기화 op의 run() 메서드를 이용해서 'x'를 초기화합시다.
# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# 'x'에서 'a'를 빼는 작업을 추가하고 실행시켜서 결과를 봅시다.
# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# 실행을 마치면 Session을 닫읍시다.
# Close the Session when we're done.
sess.close()
```

## Tensors

TensorFlow 프로그램은 모든 데이터를 tensor 데이터 구조로 나타냅니다. 연산 graph에 있는 작업들(op) 간에는 tensor만 주고받을 수 있기 때문입니다. TensorFlow의 tensor를 n 차원의 배열이나 리스트라고 봐도 좋습니다. tensor는 정적인 타입(static type), 차원(rank 역자 주: 예를 들어 1차원, 2차원하는 차원), 형태(shape, 역자 주: 예를 들어 2차원이면 m x n) 값을 가집니다. TensorFlow가 어떻게 이 개념들을 다루는지 알고 싶으시면 [Rank, Shape, and Type](../resources/dims_types.md)을 참조하시기 바랍니다.

## Variables

그래프를 실행하더라도 변수(variable)의 상태는 유지됩니다. 아래에서 간단한 카운터 예제를 통해 변수에 대해서 알 수 있습니다. 더 자세한 내용은 [Variables](../how_tos/variables/index.md)를 참조하시기 바랍니다.

```python
# 값이 0인 스칼라로 초기화된 변수를 만듭니다.
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# 'state'에 1을 더하는 작업(op)을 만듭니다.
# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 그래프를 한 번 작동시킨 후에는 'init' 작업(op)을 실행해서 변수를 초기화해야
# 합니다. 먼저 'init' 작업(op)을 추가해 봅시다.
# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# graph와 작업(op)들을 실행시킵니다.
# Launch the graph and run the ops.
with tf.Session() as sess:
  # 'init' 작업(op)을 실행합니다.
  # Run the 'init' op
  sess.run(init_op)
  # 'state'의 시작값을 출력합니다.
  # Print the initial value of 'state'
  print(sess.run(state))
  # 'state'값을 업데이트하고 출력하는 작업(op)을 실행합니다.
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

이 코드에서 `assign()` 작업은 `add()` 작업처럼 graph의 한 부분입니다. 그래서 `run()`이 graph를 실행시킬 때까지 실제로 작동하지 않습니다.

우리는 보통 통계 모델의 파라미터를 변수로 표현합니다. 예를 들어 뉴럴 네트워크의 비중값을 변수인 tensor로 표현할 수 있습니다. 학습을 진행할 때 훈련용 graph를 반복해서 실행시키고 이 tensor 값을 업데이트하는 것입니다.

## Fetches

작업의 결과를 가져오기 위해 `Session` 오브젝트에서 `run()`을 호출해서 graph를 실행하고 tensor로 결과값을 끌어냅니다. 앞의 예제에서는 'state' 하나의 노드만 가져왔지만 복수의 tensor를 받아올 수도 있습니다.:

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

여러 tensor들의 값을 계산해내기 위해 수행되는 작업(op)들은 각 tensor 별로 각각 수행 되는 것이 아니라 전체적으로 한 번만 수행됩니다.

## Feeds

위의 예제에서 살펴본 graph에서 tensor들은 `상수(Constant)` 와 `변수(Variable)`로 저장되었습니다. TensorFlow에서는 graph의 연산에게 직접 tensor 값을 줄 수 있는 'feed 메커니즘'도 제공합니다.

feed 값은 일시적으로 연산의 출력값을 입력한 tensor 값으로 대체합니다. feed 데이터는 `run()`으로 전달되어서 `run()`의 변수로만 사용됩니다. 가장 일반적인 사용방법은 tf.placeholder()를 사용해서 특정 작업(op)을 "feed" 작업으로 지정해 주는 것입니다.

```python

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]
```

만약 feed 를 제대로 제공하지 않으면 `placeholder()` 연산은 에러를 출력할 것입니다. feed를 이용하는 다른 예시는 [MNIST fully-connected feed tutorial](../tutorials/mnist/tf/index.md)([source code](https://www.tensorflow.org/code/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py))를 참조하시기 바랍니다.