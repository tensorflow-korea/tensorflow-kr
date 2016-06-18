# 변수: 생성, 초기화, 저장, 복구

모델을 학습시키는 경우에, 파라미터를 저장하고 업데이트 하기 위하여 변수들([variables](../../api_docs/python/state_ops.md))를 사용합니다. 변수들은 텐서들을 저장하는 메모리상의 버퍼들입니다. 명확하게 초기화되어야 하며, 학습이 진행되는 동안이나 끝난 후에 디스크에 저장될 수 있습니다. 나중에 모델을 사용하거나 분석하기 위하여 저장된 값들을 복구할 수 있습니다.

이 문서는 다음의 텐서플로우 클래스들을 이용하여 설명을 진행합니다. API에 대한 완전한 설명을 담고 있는 레퍼런스 매뉴얼을 보려면 링크를 참조하십시요.

*  [`tf.Variable`](../../api_docs/python/state_ops.md#Variable) 클래스.
*  [`tf.train.Saver`](../../api_docs/python/state_ops.md#Saver) 클래스.


## 생성

변수([Variable](../../api_docs/python/state_ops.md))를 생성하는 경우, 초기값으로서 `Tensor`값을 `Variable()` 생성자에 전달하게 됩니다. 텐서플로우가 제공하는 여러가지 연산자들인 [constants or random values](../../api_docs/python/constant_op.md)을 이용하여 초기화에 사용되는 텐서들을 만드는 경우가 많습니다.

이 연산자들을 사용할 때에는, 텐서의 형태를 지정해 주어야 합니다. 그렇게 지정된 텐서의 형태가 자동적으로 해당 변수의 형태가 됩니다. 변수들은 일반적으로 고정된 형태를 갖습니다만, 텐서플로우에서는 변수의 형태를 변경할 수 있는 고급 기능도 제공됩니다.

```python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
```

`tf.Variable()`를 호출하면 다음 몇가지의 연산자가 그래프에 추가됩니다:

* 변수 값을 저장하는 `variable` 연산자.
* 변수를 초기값으로 세팅하는 초기화 연산자인 `tf.assign`.
* 초기값을 위한 연산자들. 예제에 설명된 `biases` 변수를 위한 `zeros` 연산자 같은 것도 그래프에 추가됨. 

`tf.Variable()` 가 리턴하는 값은 파이썬 클래스 `tf.Variable`의 인스턴스입니다.

### 디바이스 지정

변수가 생성될 때, 특정 디바이스를 사용하도록 지정할 수 있습니다. [`with tf.device(...):`](../../api_docs/python/framework.md#device) 블럭을 사용하십시요:

```python
# Pin a variable to CPU.
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# Pin a variable to GPU.
with tf.device("/gpu:0"):
  v = tf.Variable(...)

# Pin a variable to a particular parameter server task.
with tf.device("/job:ps/task:7"):
  v = tf.Variable(...)
```

**주의** [`v.assign()`](../../api_docs/python/state.md#Variable.assign) 처럼 변수를 변경(mutate)하는 오퍼레이션과, [`tf.train.Optimizer`](../../api_docs/python/train.md#Optimizer) 의 파라미터 업데이트 오퍼레이션들은, *반드시* 변수들과 같은 디바이스상에서 실행되어야 합니다. 위의 오퍼레이션들을 생성할 때, 잘못된 디바이스 지시자(directive)는 무시됩니다.  

디바이스를 정확히 지정하는 것은 복제된 세팅에서 실행하는 경우에 특히 중요합니다. 복제된 세팅에서 디바이스 설정을 심플하게 해주는 함수들에 대해서는 [`tf.train.replica_device_setter()`](../../api_docs/python/train.md#replica_device_setter)를 참조하십시요.

## 초기화

변수의 초기화는 모델상의 어떤 오퍼레이션보다도 분명하게 먼저 실행되어야 합니다. 가장 쉬운 방법은 모든 변수의 초기화 동작을 모아서 실행하는 오퍼레이션을 만들어서, 모델을 사용하기 전에 그 오퍼레이션을 실행하는 것입니다.

또는 체크포인트 화일로부터 변수의 값을 복구하는 방법을 사용할 수도 있습니다. 아래를 참조하십시요.

`tf.initialize_all_variables()` 를 사용하여 변수들의 초기화를 실행하는 오퍼레이션을 추가합니다. 모델을 완전하게 구축하고서 세션에서 띄운(launch) 후에 그 오퍼레이션을 실행시키십시요.

```python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Later, when launching the model
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)
  ...
  # Use the model
  ...
```

### 다른 변수값을 참조하여 초기화 하기

다른 변수의 값을 이용하여 어떤 변수를 초기화해야 하는 경우가 있습니다. `tf.initialize_all_variables()` 를 통해서 추가된 op는 모든 변수들을 한꺼번에 병렬적으로 초기화 하므로, 이 기능을 이용하는 경우에는 주의해야 합니다.

다른 변수의 값을 이용하여 새로운 변수를 초기화하는 경우, 그 다른 변수의 `initialized_value()` 프로퍼티를 사용합니다. 그 초기화된 값을 그대로 새로운 변수의 초기값으로 사용하거나, 초기값을 계산하는 데에 필요한 텐서로 사용할 수 있습니다.

```python
# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
```

### 커스텀 초기화

컨비니언스 함수인 `tf.initialize_all_variables()` 는 모델상의 *모든 변수*를 초기화 하는 op를 추가합니다. 초기화가 필요한 변수들만을 리스트로 넘기는 것도 가능합니다. 변수들이 초기화 되어있는지를 체크하는 방법을 포함한 자세한 옵션들에 대해서는 [변수](../../api_docs/python/state_ops.md) 문서를 참조하십시요. 

## 저장과 복구

모델을 저장하고 복구하는 가장 쉬운 방법은 `tf.train.Saver` 객체를 사용하는 것입니다. `save` 와 `restore` op 들이 그래프에 추가되어, 모든 변수들 또는 리스트로 정리된 특정 변수들을 저장하고 복구할 수 있게 됩니다. 체크포인트 화일들을 읽고 쓸 경로를 지정하여 이 op 들을 실행시키는 메쏘드는 saver 객체가 제공합니다.

### 체크포인트 화일

변수들은 바이너리 화일에 저장되는데, 대략적으로 변수명과 텐서 값을 매핑하는 구조로 되어 있습니다.

`Saver` 객체를 생성할 때, 체크포인트 화일에 들어갈 변수들의 이름을 지정하는 것도 가능합니다. 특별히 지정하지 않는 경우에는, 각 변수들의 [`Variable.name`](../../api_docs/python/state_ops.md#Variable.name) 프로퍼티 값들이 사용됩니다.

[`inspect_checkpoint`](https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py) 라이브러리, 좀 더 자세하게는 `print_tensors_in_checkpoint_file` 함수를 사용하여, 체크포인트에 어떤 변수들이 들어있는지를 알아볼 수 있습니다.

### 변수 저장

모델안의 모든 변수를 관리하기 위하여, `tf.train.Saver()`를 사용해 `Saver`를 만드십시요.


```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)
```

### 변수 복구

변수를 복구하는 데에 위의 `Saver` 객체가 사용됩니다. 화일로부터 변수들을 복구하는 경우에는, 그 변수들을 미리 초기화 할 필요가 없습니다.


```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```

### 저장 및 복구할 변수들의 선택

If you do not pass any argument to `tf.train.Saver()` the saver handles all
variables in the graph.  Each one of them is saved under the name that was
passed when the variable was created.

It is sometimes useful to explicitly specify names for variables in the
checkpoint files.  For example, you may have trained a model with a variable
named `"weights"` whose value you want to restore in a new variable named
`"params"`.

It is also sometimes useful to only save or restore a subset of the variables
used by a model.  For example, you may have trained a neural net with 5 layers,
and you now want to train a new model with 6 layers, restoring the parameters
from the 5 layers of the previously trained model into the first 5 layers of
the new model.

You can easily specify the names and variables to save by passing to the
`tf.train.Saver()` constructor a Python dictionary: keys are the
names to use, values are the variables to manage.

Notes:

*  You can create as many saver objects as you want if you need to save and
   restore different subsets of the model variables.  The same variable can be
   listed in multiple saver objects, its value is only changed when the saver
   `restore()` method is run.

*  If you only restore a subset of the model variables at the start
   of a session, you have to run an initialize op for the other variables.  See
   [`tf.initialize_variables()`](../../api_docs/python/state_ops.md#initialize_variables)
   for more information.

```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...
```
