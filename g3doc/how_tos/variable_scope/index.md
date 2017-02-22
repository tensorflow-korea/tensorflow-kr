# 변수 공유

[Variables HowTo](../../how_tos/variables/index.md)에 
설명된 방법으로 단일 변수를 생성, 초기화, 저장 및 불러오기를 할 수 있습니다.
하지만 복잡한 모델을 구축할 때는 가끔 큰 변수 세트를 공유할 필요가 있고 한 곳에서 모든 변수의 초기화를 해야 할 수도 있습니다. 
이번 튜토리얼은 `tf.variable_scope()`와 `tf.get_variable()`를 사용해서 이것을 어떻게 할 수 있는지 보여줍니다. 

## 문제점

우리의 [Convolutional Neural Networks Tutorial](../../tutorials/deep_cnn/index.md) 모델과 유사하지만
2개의 콘볼루션(예제의 간단함을 위해)만 사용하는 이미지 필터에 대한 간단한 모델을 만든다고 가정하십시오.
[Variables HowTo](../../how_tos/variables/index.md)의 설명대로 `tf.Variable`을 바르게 사용했다면 여러분의 모델은 아래와 같을 겁니다.

```python
def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)
```

여러분이 쉽게 상상할 수 있듯이, 모델은 이것보다 훨씬 더 복잡하며, 여기에도 이미 4개의 다른 변수가 있습니다:
`conv1_weights`, `conv1_biases`, `conv2_weights`, 그리고 `conv2_biases`.

문제는 이 모델을 다시 사용하고자 할 때 발생합니다. 
2개의 다른 이미지, `image1`과 `image2`를 여러분의 이미지 필터에 적용하기를 원한다고 가정하십시오.
여러분은 같은 파라미터로 같은 필터에서 처리된 이미지가 필요합니다.
`my_image_filter()`를 두 번 호출할 수 있지만, 이것은 두 세트의 변수를 생성합니다 :

```python
# First call creates one set of variables.
result1 = my_image_filter(image1)
# Another set is created in the second call.
result2 = my_image_filter(image2)
```

변수를 공유하는 일반적인 방법은 별도의 코드로 작성하여 이를 사용하는 함수에 전달하는 것입니다. dictionary를 사용하는 예를 들어 봅시다:

```python
variables_dict = {
    "conv1_weights": tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    "conv1_biases": tf.Variable(tf.zeros([32]), name="conv1_biases")
    ... etc. ...
}

def my_image_filter(input_images, variables_dict):
    conv1 = tf.nn.conv2d(input_images, variables_dict["conv1_weights"],
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + variables_dict["conv1_biases"])

    conv2 = tf.nn.conv2d(relu1, variables_dict["conv2_weights"],
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + variables_dict["conv2_biases"])

# The 2 calls to my_image_filter() now use the same variables
result1 = my_image_filter(image1, variables_dict)
result2 = my_image_filter(image2, variables_dict)
```

위와 같은 변수 생성은 편리하지만, 코드 밖에서 캡슐화가 중단됩니다.

*  그래프를 빌드하는 코드는 생성할 변수의 이름, 타입, 그리고 모양(shape)을 반드시 기록해야 합니다.
*  코드가 변경되면 호출자(callers)는 어느 정도 다른 변수를 생성해야 할지도 모릅니다.

문제를 해결하는 한 가지 방법은 모델을 생성할 때 필요한 변수를 관리하는 클래스(classes)를 사용하는 것입니다.
TensorFlow는 클래스를 포함하지 않는 더 가벼운 솔루션을 위해 그래프를 구성하는 동안 명명된 변수를 쉽게 공유할 수 있는 *Variable Scope* 메커니즘을 제공합니다.

## 변수 범위(Variable scope) 예제

TensorFlow의 Variable Scope 메커니즘은 두 개의 메인 함수로 되어 있습니다.

* `tf.get_variable(<name>, <shape>, <initializer>)`:
  입력된 이름의 변수를 생성하거나 반환합니다.
* `tf.variable_scope(<scope_name>)`:
  Manages namespaces for names passed to `tf.get_variable()`.
  `tf.get_variable()`에 전달 된 이름의 네임스페이스를 관리합니다.

`tf.get_variable()`함수는 `tf.Variable`을 직접호출 대신 변수를 가져오거나 생성하는 데 사용합니다.
`tf.Variable`처럼 직접 값을 전달하는 대신 *initializer*를 사용합니다. initializer는 모양(shape)을 가져와서 텐서를 제공하는 함수입니다. 
여기 TensorFlow에서 사용 가능한 몇 개의 initializer가 있습니다.

* `tf.constant_initializer(value)` 제공된 값으로 모든 것을 초기화합니다,
* `tf.random_uniform_initializer(a, b)` [a, b]를 균일하게 초기화 합니다,
* `tf.random_normal_initializer(mean, stddev)` 주어진 평균 및 표준 편차로 정규 분포에서 초기화합니다.

`tf.get_variable()`이 앞에서 논의한 문제를 어떻게 해결하는지 보시려면, 하나의 convolution을 생성한 코드를 `conv_relu`라는 별개의 함수로 리펙토링합시다:

```python
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```

이 함수는 짧은 이름 `"weights"`와 `"biases"`를 사용합니다.
우리는 그것을 `conv1`과`conv2` 둘 다에서 사용하기를 원하지만, 변수들은 다른 이름을 가질 필요가 있습니다.
이곳은`tf.variable_scope()`가 동작하는 곳입니다 :
변수에 대한 네임 스페이스를 푸시(pushes)합니다.

```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

이제, 우리가 `my_image_filter()`를 두 번 호출할 때 벌어지는 일을 봅시다.

```
result1 = my_image_filter(image1)
result2 = my_image_filter(image2)
# Raises ValueError(... conv1/weights already exists ...)
```

보다시피, `tf.get_variable()`은 이미 존재하는 변수가 우연히 공유된 것이 아닌지 확인합니다.
만약 공유하기를 원하면, 여러분은 다음과 같이 `reuse_variables()`를 설정해야 합니다.

```
with tf.variable_scope("image_filters") as scope:
    result1 = my_image_filter(image1)
    scope.reuse_variables()
    result2 = my_image_filter(image2)
```

이것은 가볍고 안전하게 변수를 공유하는 좋은 방법입니다.

## 변수 범위는 어떻게 동작합니까?

### `tf.get_variable()` 이해하기

변수 범위를 이해하기 위해서는 첫 번째로 `tf.get_variable()`이 어떻게 동작하는지 완전히 이해하는 것이 필요합니다.
다음은`tf.get_variable()`가 일반적으로 호출되는 방법입니다.

```python
v = tf.get_variable(name, shape, dtype, initializer)
```

호출되는 범위에 따라 두 가지 중 하나를 호출합니다.
다음은 두 가지 옵션입니다.

* Case 1: 범위는 `tf.get_variable_scope().reuse == False`에서 증명된 것처럼 새로운 변수를 생성하기 위해 설정됩니다.

이 경우, `v`는 제공된 모양(shape)과 데이터 타입을 가지는 새롭게 만들어진`tf.Variable`이 됩니다.
생성된 변수의 전체 이름은 현재 변수 범위 이름 + 제공된`name`으로 설정되고, 이 전체 이름을 가진 변수가 아직 존재하지 않는지 보장하기 위한 검사를 하게 됩니다.
이 전체 이름이 변수로 사용 중이라면, 함수는 `ValueError`를 발생시킵니다.
만약 새로운 변수를 생성하면, 변수는 `initializer(shape)` 값으로 초기화될 것입니다. 예를 들어:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
```

* Case 2: `tf.get_variable_scope().reuse == True`에서 증명된 것처럼 변수를 재사용하기 위한 범위가 설정됩니다.

이 경우, 호출은 현재 변수 범위 이름 + 제공된`name`과 같은 이름으로 이미 존재하는 변수를 검색합니다.
만약 변수가 없으면 `ValueError`가 발생할 겁니다. 변수가 발견된다면 반환됩니다. 예를 들어:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

### `tf.variable_scope()`의 기본

`tf.get_variable()`가 어떻게 동작하는지 알면 변수 범위를 쉽게 이해하고 만들 수 있습니다.
변수 범위의 주요 기능은 변수 이름에 대한 접두사로 사용되는 이름을 들고 있는 것이고, 위에서 설명한 두 가지 경우를 구별하기 위한 재사용-플래그입니다.
중첩 변수 범위는 디렉터리가 동작하는 방식과 유사한 방법으로 그것들의 이름을 추가합니다.

```python
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
assert v.name == "foo/bar/v:0"
```

현재 변수 범위는 `tf.get_variable_scope()`를 사용해서 회수할 수 있으며, 
현재 변수 범위의 `reuse` 플래그는 `tf.get_variable_scope().reuse_variables()`를 호출해서 `True`로 설정할 수 있습니다.

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

여러분은 `reuse` 플래그를 `False`로 설정할 수는 없습니다. 
이유는 모델을 생성하는 함수를 구성하는 것을 허용하기 위해서입니다.
이전처럼`my_image_filter(inputs)` 함수를 작성한다고 상상해보십시오. 
변수 범위에서 `reuse=True`와 함께 함수를 호출하는 누군가는 모든 내부 변수가 재사용 될 것으로 예상합니다.
함수 내부에서 `reuse=False`를 강제로 허용하면 이 계약이 깨지게 되고, 이런 방법으로 파라미터를 공유하는 것을 어렵게 만듭니다.

`reuse`를 명시적으로 `False`로 설정할 수는 없지만, 재사용 가능한 변수 범위를 입력한 다음 종료하고 재사용하지 않는 변수 범위로 돌아갈 수 있습니다.
변수 범위를 열 때`reuse = True` 파라미터를 사용하여 이 작업을 수행할 수 있습니다.
또한, 위와 같은 이유로`reuse` 파라미터가 상속된다는 점에 유의하십시오. 따라서 재사용 가능한 변수 범위를 열면 모든 하위 범위(sub-scopes)도 재사용됩니다.

```python
with tf.variable_scope("root"):
    # At start, the scope is not reusing.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo", reuse=True):
        # Explicitly opened a reusing scope.
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
            assert tf.get_variable_scope().reuse == True
    # Exited the reusing scope, back to a non-reusing one.
    assert tf.get_variable_scope().reuse == False
```

### 변수 범위 캡처(Capturing)

위에 제시된 모든 예제에서, 우리는 변수들의 이름이 일치했기 때문에 파라미터를 공유할 수 있었는데, 
그건, 정확히 같은 문자열로 재사용 변수 범위를 열었기 때문입니다.
더 복잡한 경우에는 이름을 올바르게 가져와서 전달하는 것보다 VariableScope 객체를 통과시키는 것이 유용할 수도 있습니다.
이를 위해, 변수 범위는 새로운 변수 범위를 열 때 이름 대신 캡처해서 사용할 수 있습니다.


```python
with tf.variable_scope("foo") as foo_scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope)
    w = tf.get_variable("w", [1])
with tf.variable_scope(foo_scope, reuse=True)
    v1 = tf.get_variable("v", [1])
    w1 = tf.get_variable("w", [1])
assert v1 == v
assert w1 == w
```

변수 범위를 열 때 이전에 존재하는 범위를 사용하면 현재 변수 범위 접두사에서 완전히 다른 곳으로 넘어갑니다.
이것은 우리가 하는 곳과 완전히 독립적입니다.

```python
with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"
with tf.variable_scope("bar")
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"  # Not changed.
```

### 변수 범위의 Initializers

`tf.get_variable()`을 사용하면 변수를 생성하거나 재사용하는 함수를 작성하는 것이 허용되며 외부에서 투명하게 호출할 수 있습니다.
하지만 생성된 변수의 initializer를 변경하려면 어떻게 해야 할까요?
변수를 만드는 모든 함수에 추가인수를 전달하는 것이 필요할까요?
모든 변수의 기본 initializer를 모든 함수의 위의 한 곳에서 설정하기를 원할 때 일반적인 때는 어떻습니까?
이런 경우를 돕기 위해, 변수 범위는 기본 initializer를 들고 있을 수 있습니다.
하위 범위(sub-scopes)에 의해 상속받고 각각 `tf.get_variable()`호출에 전달됩니다.
그러나 다른 initializer를 명시적으로 지정하면 오버라이드(overridden) 될 겁니다.

```python
with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
    v = tf.get_variable("v", [1])
    assert v.eval() == 0.4  # Default initializer as set above.
    w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3)):
    assert w.eval() == 0.3  # Specific initializer overrides the default.
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.eval() == 0.4  # Inherited default initializer.
    with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):
        v = tf.get_variable("v", [1])
        assert v.eval() == 0.2  # Changed default initializer.
```

### `tf.variable_scope()` 오퍼레이션(ops)의 이름

우리는 어떻게 `tf.variable_scope`가 변수들의 이름을 관리하는지 논의했습니다.
하지만 그것이 범위 내에서 다른 오퍼레이션(ops)의 이름에 영향을 미치나요?
변수 범위 내에서 생성된 오퍼레이션(ops)은 이름이 공유되는 것이 정상적입니다.
이런 이유로, 우리가 `with tf.variable_scope("name")`를 사용할 때 이것은 무조건 `tf.name_scope("name")`를 엽니다.
예를 들어:

```python
with tf.variable_scope("foo"):
    x = 1.0 + tf.get_variable("v", [1])
assert x.op.name == "foo/add"
```

이름 변수는 변수 범위를 추가로 열 수 있으며, 그것들은 변수가 아닌 오직 오퍼레이션(ops)의 이름에만 영향을 미칩니다. 

```python
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
assert v.name == "foo/v:0"
assert x.op.name == "foo/bar/add"
```

문자열 대신 캡처된 객체를 사용해서 변수 범위를 열 때,
우리는 오퍼레이션(ops)의 현재 이름 범위를 바꾸지 않습니다. 


## 사용 예시

다음은 변수 범위를 사용하는 몇 가지 파일에 대한 포인터(pointers)입니다.
특히, RNN(recurrent neural networks)과 seq2seq(sequence-to-sequence) 모델에서 많이 사용됩니다.

파일 | 내용 
--- | ---
`models/image/cifar10.py` | 이미지 내에서 객체를 찾는 모델.
`models/rnn/rnn_cell.py` | RNN을 위한 Cell 함수.
`models/rnn/seq2seq.py` | seq2seq 모델을 구축하기 위한 함수.
