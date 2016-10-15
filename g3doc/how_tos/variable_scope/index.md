# 변수 공유

여러분은 다음과 같은 방법으로 변수를 생성, 초기화, 저장, 복구할 수 있습니다.
[Variables HowTo](../../how_tos/variables/index.md).
하지만 더욱 복잡한 모델을 설계할 때에, 여러분은 종종 거대한 변수 set을 공유하거나,
해당 변수들을 한곳에서 초기화할 필요가 있습니다.
이 튜토리얼은 여러분이 `tf.variable_scope()`와 `tf.get_variable()`를 어떻게 사용하는 지 보여줍니다.

## 문제

이미지 필터를 위한 간단한 모델을 하나 만든다고 생각해봅시다.
[Convolutional Neural Networks Tutorial](../../tutorials/deep_cnn/index.md)
모델과 비슷하지만 (예시를 간단하게 하기 위해) 오직 2개의 convolution을 갖는 모델일 때, 여러분이
[Variables HowTo](../../how_tos/variables/index.md)에서 봤던 `tf.Variable`만을 사용한다면,
모델은 이렇게 만들어질 것입니다.

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

여러분의 생각과 같이, 모델은 이보다 더 복잡해질 것입니다.
게다가 우리는 이미 4개의 다른 변수: `conv1_weights`, `conv1_biases`, `conv2_weights`, `conv2_biases`를 가지고있습니다

이제 문제는 우리가 이 모델을 재사용할 때 발생합니다.
이미지 필터를 두개의 다른 이미지 `image1`와 `image2`에 적용한다고 가정합시다.
두 이미지에 같은 필터로 같은 parameters값을 적용하고 싶다면,
다음과 같이 `my_image_filter()` 를 두번 호출하면 됩니다. 다만 이것은 두 개의 변수 set을 생성하게 될 것입니다:

```python
# First call creates one set of variables.
result1 = my_image_filter(image1)
# Another set is created in the second call.
result2 = my_image_filter(image2)
```

변수를 공유하는 가장 일반적인 방법은 변수 선언을 다른 조각으로 분리하고, 함수에 전달하여 사용하는것입니다.
dictionary를 사용한 예:

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

변수들을 위처럼 코드 바깥에 만들면, 편리한 반면에 encapsulation 속성을 해치게 됩니다:

*  그래프를 그리기 위한 코드는 생성할 변수의 이름, 타입, 형태(shape)를 기재해야 합니다.
*  코드가 변하면, 호출자는 더 많거나, 적거나 또는 다른 변수들을 생성해야 될 수도 있습니다.

이를 해결하기 위한 하나의 방법은, 모델을 만드는 클래스를 사용하고, 클래스에서 필요한 변수들을 관리하는 것입니다.
클래스를 사용하지 않는 좀 더 가벼운 해결방법을 위해 텐서플로우는 그래프를 그릴 수 있는 동시에, 이름을 갖는 변수들을 공유가능한
*변수 scope(Variable Scope)* 메커니즘을 제공합니다.

## 변수 scope 예제

텐서플로우의 변수 scope 메커니즘은 2가지 main function을 가집니다:

* `tf.get_variable(<name>, <shape>, <initializer>)`:
  주어진 이름의 변수를 만들거나 리턴합니다.
* `tf.variable_scope(<scope_name>)`:
  `tf.get_variable()`에 전달됐던 이름들에 대한 namespace들을 관리합니다.

함수 `tf.get_variable()`는 `tf.Variable`를 직접 호출하는 대신 변수를 생성하거나 얻는데에 쓰입니다.
해당 함수는 변수를 직접 전달하는 대신 *initializer* 를 이용합니다. `tf.Variable`에서와 같이,
initializer는 형태(shape)를 취해 텐서와 형태(shape)를 제공하는 함수입니다.
텐서플로우에서 이용가능한 initializer들을 아래에서 확인하실 수 있습니다:

* `tf.constant_initializer(value)` 제공된 값으로 모든것들을 초기화합니다,
* `tf.random_uniform_initializer(a, b)` a와 b를 포함하는 사이의 값([a, b])들 중에서 균등한 분포의 임의값들로 초기화합니다,
* `tf.random_normal_initializer(mean, stddev)` 주어진 평균값과 표준편차를 가지는 정규분포에서, 이를 따르는 임의값들로 초기화합니다.

전에 이야기한 문제를 `tf.get_variable()`를 통해 어떻게 해결하는지 보기 위해,
코드를 하나의 convolution을 만드는 분리된 함수 `conv_relu`로 refactor 해봅시다:

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

이 함수는 변수들의 이름으로 간결하게 `"weights"` 와 `"biases"`를 사용합니다.
우리는 이것을 `conv1`과 `conv2`에 사용하고싶지만, 각각의 변수들은 다른 이름을 가져야 합니다.
자, 이제 `tf.variable_scope()`가 역할을 할 차례입니다:
이것은 변수들이 namespace를 갖도록 해줍니다.

```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

자, 이제 우리가 `my_image_filter()`를 두번 호출했을 때 무슨 일이 일어나는지 봅시다.

```
result1 = my_image_filter(image1)
result2 = my_image_filter(image2)
# Raises ValueError(... conv1/weights already exists ...)
```

보시다시피, `tf.get_variable()` 는 이미 존재하는 변수들이 우연히라도 공유되지 않도록 검사합니다.
만약 여러분이 변수를 공유하고 싶다면, 다음과 같이 `reuse_variables()`를 통해 명시해야 합니다.

```
with tf.variable_scope("image_filters") as scope:
    result1 = my_image_filter(image1)
    scope.reuse_variables()
    result2 = my_image_filter(image2)
```

이것이 변수들을 공유하는, 간단하고 안전한 방법입니다.

## 변수 scope는 어떻게 동작하나요?

### `tf.get_variable()`에 대한 이해

변수 scope를 이해하기 위해 먼저 `tf.get_variable()`가 어떻게 동작하는지 완벽히 이해해야 합니다.
아래는 `tf.get_variable`가 호출되는 일반적인 경우입니다.

```python
v = tf.get_variable(name, shape, dtype, initializer)
```

이 경우 함수 호출이 포함된 스코프에 따라 둘 중 하나를 실행합니다.
여기 두 가지 경우가 있습니다.

* 경우 1: `tf.get_variable_scope().reuse == False`에 따라서 scope가 새 변수를 만들도록 설정되어 있을 경우.

이 경우, `v`는 제공된 형태(shape)와 데이터 타입에 따라 새로 생성된 `tf.Variable`입니다.
생성된 변수의 완전한 이름은 (현재 scope 이름 + 제공된 `name`)가 되고,
이미 존재하는 변수의 이름과 겹치지 않는지 검증하는 과정이 이루어집니다.
만약 이름이 일치하는 변수가 이미 존재한다면, 함수는 `ValueError`를 발생시킵니다.
새로운 변수가 생성된다면 변수는 `initializer(shape)`의 값으로 초기화됩니다. 예시:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
```

* 경우 2: `tf.get_variable_scope().reuse == True`에 따라서 scope가 변수를 재사용하도록 설정되어 있을 경우.

이 경우, 호출은 (현재 scope 이름 + 제공된 `name`)을 가진 이미 존재하는 변수를 찾는 형태로 이루어집니다.
만약 해당 변수가 존재하지 않는다면, `ValueError`를 발생시킵니다. 해당 변수가 존재한다면 해당 변수를 리턴합니다. 예시:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

### `tf.variable_scope()`의 기본

`tf.get_variable()`이 어떻게 동작하는지 아는 것은 변수 scope를 이해하기 쉽게 해줍니다.
변수 scope의 주요한 기능은 변수 이름의 접두사로 붙일 이름과, 앞서 설명한 두가지 경우를 구별하기 위한 reuse-flag를 지니는 것입니다.
중첩된 변수 scope는 디렉토리와 유사한 방식으로 이름을 덧붙입니다:

```python
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
assert v.name == "foo/bar/v:0"
```

현재 변수 scope를 `tf.get_variable_scope()`를 통해 가져올 수 있습니다.
그리고 `tf.get_variable_scope().reuse_variables()`를 호출하여 현재 변수 scope의 `reuse` flag를 `True`로 설정할 수 있습니다:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

`reuse` flag를 `False`로 *설정할 수 없다* 는 것을 명심하세요.
이유는 모델을 만드는 함수를 구성하는 데에서 찾을 수 있습니다.
전처럼 `my_image_filter(inputs)`함수를 만든다고 생각해 봅시다.
`reuse=True`된 scope 안에서 그 함수를 호출하는 누군가는 함수 내의 변수 또한 재사용되기를 바랄 수 있습니다.
함수 내에서 `reuse=False`를 허락하는 것은 이 약속을 깨버릴 뿐더러 파라미터의 공유 또한 어렵게 만들어 버립니다.

`reuse`를 `False`로 명시적으로 설정할 수 없더라도 여러분은 변수를 재사용하는 scope에 들어왔다 나가서,
재사용하지 않는 scope로 돌아갈 수 있습니다.
이는 scope를 설정할 때, `reuse=True` 파라미터를 사용함으로써 가능합니다.
또한 위와 같은 이유로 `reuse` 파라미터는 상속된다는 것을 명심하세요. 따라서 한번 변수를 재사용하는 scope를 설정헀다면,
해당 scope의 모든 sub-scope는 변수를 재사용하게 됩니다.

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

### scope 가져오기

위의 모든 예시에선, 우리는 같은 이름으로만 변수들을 재사용 할 수 있었습니다.
왜냐하면, 변수를 재사용하는 scope를 설정할 때 해당 스코프가 무엇인지를 string으로만 명시해주었기 때문입니다.
여러 복잡한 상황에선 scope들의 이름에만 의지하는 것보단 VariableScope 객체 자체를 전달하는 것이 더 유용할 수 있습니다.
이를 위해, 변수 scope를 설정할 때에 이름 대신 VariableScope 객체를 직접 전달할 수 있습니다.

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

이미 존재하는 scope 객체를 이용하여 스코프를 설정할 때,
호출한 위치의 바깥에 설정되어있는 scope에 상관 없이 바로 다른 scope로 넘어갈 수 있습니다.

```python
with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"
with tf.variable_scope("bar")
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"  # Not changed.
```

### 변수 scope에 initializer 설정하기

`tf.get_variable()`를 사용하면 함수에서 변수들을 생성하고 재사용할 수 있게 하며, 이를 함수 외부에서 관여할 수 있게 해줍니다.
하지만 생성된 변수들의 initializer를 바꾸고 싶다면 어떻게 해야할까요?
추가적인 변수를 생성하는 argument를 모든 함수에 전달해야 할까요?
일반적인 경우는 어떤가요, 모든 변수의 default initializer를 모든 함수 호출 전의 한곳에서 설정하고 싶다면?
이런 경우를 위해, 변수 scope는 default initializer를 가질 수 있습니다.
이것은 sub-scope에도 상속되어 각각의 `tf.get_variable()` 호출에 전달됩니다.
다만 명시된 다른 initializer를 통해 override 할 수 있습니다.

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

### `tf.variable_scope()` 안의 ops들의 이름

지금까지 `tf.variable_scope`가 어떻게 변수들의 이름을 관리하는 지 알아보았습니다.
그렇다면 이것이 scope 안의 ops들에 어떤 영향을 미칠까요?
변수 scope 안에 생성된 ops들 또한 scope의 이름을 공유하는것이 자연스럽기에,
`with tf.variable_scope("name")`를 설정한다면 암묵적으로 `tf.name_scope("name")` 또한 설정됩니다. 예시:

```python
with tf.variable_scope("foo"):
    x = 1.0 + tf.get_variable("v", [1])
assert x.op.name == "foo/add"
```

또한 이름 scope를 변수 scope내에 설정할 수 있습니다.
이것은 오직 ops의 이름에만 영향을 주며, 변수에는 영향을 주지 않습니다.

```python
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
assert v.name == "foo/v:0"
assert x.op.name == "foo/bar/add"
```

string이 아닌 scope 객체를 전달하여 scope를 설정할 때엔, 이름 scope는 바뀌지 않습니다.


## Examples of Use

여기에 변수 scope를 사용한 예시가 있습니다.
recurrent neural networks와 sequence-to-sequence models에는 특별히 많이 사용되었습니다.

파일 이름 | 내용
--- | ---
`models/image/cifar10.py` | 이미지 속 물체를 인지하는 모델.
`models/rnn/rnn_cell.py` | recurrent neural networks를 위한 Cell function.
`models/rnn/seq2seq.py` | building sequence-to-sequence model 설계를 위한 Functions.
