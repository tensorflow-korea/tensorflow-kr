<!-- This file is machine generated: DO NOT EDIT! -->

# 상수, 시퀀스, 그리고 난수

참고 : `Tensor`를 인자로 받는 함수들은 [`tf.convert_to_tensor`](framework.md#convert_to_tensor)의 인자로 들어갈 수 있는 값들 또한 받을 수 있습니다.

[TOC]

## 상수값 텐서

TensorFlow는 상수를 생성할 수 있는 몇가지 연산을 제공합니다.

- - -

### `tf.zeros(shape, dtype=tf.float32, name=None)` {#zeros}

모든 원소의 값이 0인 텐서를 생성합니다.

이 연산은 모든 원소의 값이 0이고, `shape` shape을 가진 `dtype`타입의 텐서를 반환합니다.

예시:

```python
tf.zeros([3, 4], int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

##### 인자:


*  <b>`shape`</b>: 정수 리스트 또는 `int32`타입의 1-D(1-Dimension) `Tensor`.
*  <b>`dtype`</b>: 반환되는 `Tensor`의 원소 타입.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  모든 원소의 값이 0인 `Tensor`.

- - -

### `tf.zeros_like(tensor, dtype=None, name=None)` {#zeros_like}

모든 원소의 값이 0인 텐서를 생성합니다.

하나의 텐서(`tensor`)가 주어졌을 때, 이 연산은 모든 원소의 값이 0이고 `tensor`와 같은 타입과 shape을 가진 텐서를 반환합니다. 선택적으로, `dtype`을 사용해서 새로운 타입을 지정할 수도 있습니다.

예시:

```python
# 'tensor' is [[1, 2, 3], [4, 5, 6]]
tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
```

##### 인자:


*  <b>`tensor`</b>: 하나의 `Tensor`.
*  <b>`dtype`</b>: 반환되는 `Tensor`의 타입. `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`만 가능합니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  모든 원소의 값이 0인 `Tensor`.


- - -

### `tf.ones(shape, dtype=tf.float32, name=None)` {#ones}

모든 원소의 값이 1인 텐서를 생성합니다.

이 연산은 모든 원소의 값이 1이고, `shape` shape을 가진 `dtype`타입의 텐서를 반환합니다.

예시:

```python
tf.ones([2, 3], int32) ==> [[1, 1, 1], [1, 1, 1]]
```

##### 인자:


*  <b>`shape`</b>: 정수 리스트 또는 `int32`타입의 1-D(1-Dimension) `Tensor`.
*  <b>`dtype`</b>: 반환되는 `Tensor`의 원소 타입.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  모든 원소의 값이 1인 `Tensor`.


- - -

### `tf.ones_like(tensor, dtype=None, name=None)` {#ones_like}

모든 원소의 값이 1인 텐서를 생성합니다.

하나의 텐서(`tensor`)가 주어졌을 때, 이 연산은 모든 원소의 값이 1이고 `tensor`와 같은 타입과 shape을 가진 텐서를 반환합니다. 선택적으로, `dtype`을 사용해서 새로운 타입을 지정할 수도 있습니다.

예시:

```python
# 'tensor' is [[1, 2, 3], [4, 5, 6]]
tf.zeros_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]
```

##### 인자:


*  <b>`tensor`</b>: 하나의 `Tensor`.
*  <b>`dtype`</b>: 반환되는 `Tensor`의 타입. `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`만 가능합니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  모든 원소의 값이 1인 `Tensor`.



- - -

### `tf.fill(dims, value, name=None)` {#fill}

스칼라값으로 채워진 텐서를 생성합니다.

이 연산은 `dims` shape의 텐서를 만들고 `value`로 값을 채웁니다.

예시:

```prettyprint
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

##### 인자:


*  <b>`dims`</b>: `int32`타입의 `Tensor`. 1-D(1-Dimension)이며 반환값 텐서의 shape을 나타냅니다.
*  <b>`value`</b>: 스칼라 값을 갖는 `Tensor`. 반환값 텐서에 채워지는 값입니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `value`와 같은 타입을 가진 `Tensor`.


- - -

### `tf.constant(value, dtype=None, shape=None, name='Const')` {#constant}

상수 텐서를 생성합니다.

 결과값 텐서는 `value`인자와 (선택적인) `shape`에 의해 결정됨으로써 `dtype`타입의 값으로 채워집니다. (아래 예시를 보세요.)

 인자 `value`는 상수 또는 `dtype`타입을 가진 값들의 리스트가 될 수 있습니다. 만약 `value`가 리스트라면, 리스트의 길이는 `shape`인자에 의해 나올 수 있는 원소들의 갯수와 같거나 작아야 합니다. 리스트의 길이가 `shape`에 의해 정해지는 원소들의 갯수보다 적을 경우, 리스트의 마지막 원소가 나머지 엔트리를 채우는데 사용됩니다.

 `shape`인자는 선택사항입니다. 만약 이 인자가 존재할 경우, 이는 결과값 텐서의 차원을 결정합니다. 그 외에는, `value`의 shape을 그대로 사용합니다.

 만약 `dtype`인자가 결정되지 않을 경우에는, `value`로부터 타입을 추론하여 사용합니다.

 예시:

 ```python
 # Constant 1-D Tensor populated with value list.
 tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

 # Constant 2-D tensor populated with scalar value -1.
 tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                              [-1. -1. -1.]]
 ```

##### 인자:


*  <b>`value`</b>: 반환 타입 `dtype`의 상수값 (또는 리스트).


*  <b>`dtype`</b>: 결과값 텐서 원소들의 타입.


*  <b>`shape`</b>: 결과값 텐서의 차원 (선택사항).


*  <b>`name`</b>: 텐서의 명칭 (선택사항).

##### 반환값:

  상수 `Tensor`.


## 시퀀스

- - -

### `tf.linspace(start, stop, num, name=None)` {#linspace}

구간 사이의 값들을 생성합니다.

`start`부터 시작해서 생성된 `num`개의 고르게 분포된 값들의 시퀀스입니다. 만약 `num > 1`이면, 시퀀스의 값들은 `stop - start / num - 1`씩 증가되며, 마지막 원소는 `stop`값과 같아집니다.

예시:

```
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```

##### 인자:


*  <b>`start`</b>: `float32`또는 `float64`타입의 `Tensor`. 구간의 첫번째 엔트리입니다.
*  <b>`stop`</b>: `start`와 같은 타입을 가진 `Tensor`. 구간의 마지막 엔트리입니다.
*  <b>`num`</b>: `int32`타입의 `Tensor`. 생성할 값들의 갯수입니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `start`와 같은 타입을 가진 `Tensor`. 생성된 값들은 1-D입니다.


- - -

### `tf.range(start, limit=None, delta=1, name='range')` {#range}

정수 시퀀스를 생성합니다.

`start`부터 시작하여 `limit`까지 (`limit`는 포함하지 않음) `delta`의 증가량만큼 확장하며 정수 리스트를 생성합니다.

파이썬의 내장 함수인 `range`와 유사하며, `start`의 기본값은 0이고, 즉  `range(n) = range(0, n)`입니다.

예시:

```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

# 'limit' is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]
```

##### 인자:


*  <b>`start`</b>: `int32`타입의 스칼라 값(0-D)입니다. 시퀀스의 첫번째 엔트리이며, 기본값은 0입니다.
*  <b>`limit`</b>: `int32`타입의 스칼라 값(0-D)입니다. 시퀀스의 상한이며, 시퀀스에 포함되지 않습니다. (exclusive)
*  <b>`delta`</b>: A 0-D `Tensor` (scalar) of type `int32`. Optional. Default is 1.
    Number that increments `start`. `int32`타입의 스칼라(0-D) 텐서입니다. 선택적인 인자이며, 기본값은 1입니다. `start`를 증가시키는 수입니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  1-D의 `int32`타입을 갖는 `Tensor`.


## 난수 텐서

TensorFlow는 서로 다른 분포를 가진 난수 텐서들을 생성하는 여러가지 연산들을 제공합니다. 난수 연산들은 상태를 가지며 , 계산될 때마다 새로운 난수를 생성합니다.

이러한 함수들의  `seed` 키워드 인자는 그래프 수준의 난수 시드값과 함께 작용합니다. [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 사용하는 그래프 수준의 시드 또는 연산 수준의 시드를 바꾸는 것은 이러한 연산들의 기본 시드값을 바꿀 것입니다. 연산 수준과 그래프 수준의 난수 시드 사이의 상호작용에 대해 자세히 알고 싶다면  [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 참고하십시오.

### 예시:

```python
# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

# Each time we run these ops, different results are generated
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

# Set an op-level seed to generate repeatable sequences across sessions.
norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))
```

또 다른 난수값을 사용하는 일반적인 사례는 변수들의 초기화입니다. 이 또한 [Variables How To](../../how_tos/variables/index.md)에서 볼 수 있습니다.

```python
# Use random uniform values in [0, 1) as the initializer for a variable of shape
# [2, 3]. The default type is float32.
var = tf.Variable(tf.random_uniform([2, 3]), name="var")
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
print(sess.run(var))
```

- - -

### `tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)` {#random_normal}

정규분포로부터의 난수값을 반환합니다.

##### 인자:


*  <b>`shape`</b>: 정수값의 1-D  텐서 또는 파이썬 배열. 반환값 텐서의 shape입니다.
*  <b>`mean`</b>: 0-D 텐서 또는 `dtype`타입의 파이썬 값. 정규분포의 평균값.
*  <b>`stddev`</b>: 0-D 텐서 또는 `dtype`타입의 파이썬 값. 정규분포의 표준 편차.
*  <b>`dtype`</b>: 반환값의 타입.
*  <b>`seed`</b>: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 보십시오.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  정규 난수값들로 채워진 shape으로 정해진 텐서.

- - -

### `tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)` {#truncated_normal}

절단정규분포로부터의 난수값을 반환합니다.

생성된 값들은 평균으로부터 떨어진 버려지고 재선택된 두 개의 표준편차보다 큰 값을 제외한 지정된 평균과 표준 편차를 가진 정규 분포를 따릅니다.

##### 인자:


*  <b>`shape`</b>: 정수값의 D-1 텐서 또는 파이썬 배열. 반환값 텐서의 shape입니다.
*  <b>`mean`</b>: 0-D 텐서 또는 `dtype`타입의 파이썬 값. 절단정규분포의 평균값.
*  <b>`stddev`</b>: 0-D 텐서 또는 파이썬 값. 절단정규분포의 표준 편차.
*  <b>`dtype`</b>: 반환값의 타입.
*  <b>`seed`</b>: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 보십시오.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  절단 정규 난수값들로 채워진 shape으로 정해진 텐서.

- - -

### `tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)` {#random_uniform}

균등분포로부터의 난수값을 반환합니다.

생성된 값들은 `[minval, maxval]`구간의 균등분포를 따릅니다. 하한 `minval`은 구간에 포함(included)되는 반면, 상한인 `maxval`은 포함되지 않습니다(excluded).  

실수형의 경우, 기본 구간은 `[0, 1)`입니다. 정수형의 경우, 적어도 `maxval`은 명시적으로 지정되어야합니다.

정수형의 경우, `maxval - minval`가 2의 제곱수가 아니라면 정수 난수들은 한쪽으로 약간 치우칩니다. 치우침의 정도는 `maxval - minval`의 값이 반환값의 구간(`2**32 또는 2**64`)보다 훨씬 작을 경우엔 작습니다.

##### 인자:


*  <b>`shape`</b>: 정수값의 D-1 텐서 또는 파이썬 배열. 반환값 텐서의 shape입니다.
*  <b>`minval`</b>: 0-D 텐서 또는 `dtype`타입의 파이썬 값. 난수값 생성 구간의 하한입니다. 기본값은 0입니다.
*  <b>`maxval`</b>: 0-D 텐서 또는 `dtype`타입의 파이썬 값. 난수값 생성 구간의 상한입니다. `dtype`이 실수형일 경우 기본값은 1입니다.
*  <b>`dtype`</b>: 반환값의 타입: `float32`, `float64`, `int32`, 또는 `int64`.
*  <b>`seed`</b>: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 보십시오.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  균등 난수값들로 채워진 shape으로 정해진 텐서.

##### 예외:


*  <b>`ValueError`</b>: `dtype`이 정수형인데 `maxval`이 지정되지 않을 경우 발생합니다.


- - -

### `tf.random_shuffle(value, seed=None, name=None)` {#random_shuffle}

값의 첫번째 차원을 기준으로 랜덤하게 섞어줍니다.

텐서는 0차원을 따라 섞이는데, 예를 들면 각 `value[j]`는 `output[i]`의 각 원소에 정확히 하나씩 매핑이됩니다. 예를 들면, 3x2 텐서의 경우 다음과 같은 매핑을 가질 수 있습니다.

```python
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```

##### 인자:


*  <b>`value`</b>: 섞기 위한 텐서.
*  <b>`seed`</b>: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 보십시오.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `value`의 첫번째 차원을 따라 섞인 `value`와 같은 타입과 shape을 가진 텐서.

- - -

### `tf.random_crop(value, size, seed=None, name=None)` {#random_crop}

텐서를 주어진 사이즈만큼 랜덤하게 잘라냅니다.

균등하게 선택된 오프셋에서 `value`의 일부분을 `size` shape으로 잘라냅니다. `value.shape >= size`를 만족해야합니다.

만약 차원을 잘라낼 수 없다면 차원의 전체 크기를 보냅니다.
예를 들면, RGB 이미지는 `size = [crop_height, crop_width, 3]`을 가지고 잘라낼 수 있습니다.

##### 인자:


*  <b>`value`</b>: 자르기 위한 입력 텐서.
*  <b>`size`</b>: `value`의 랭크값을 가진 1-D 텐서.
*  <b>`seed`</b>: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 보십시오.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `value`와 같은 랭크값을 갖고 `size` shape을 갖는 잘려진 텐서

- - -

### `tf.multinomial(logits, num_samples, seed=None, name=None)` {#multinomial}

다항분포로부터 샘플을 뽑아줍니다.

예시:

  samples = tf.multinomial(tf.log([[0.5, 0.5]]), 10)
  # samples has shape [1, 10], where each value is either 0 or 1.

  samples = tf.multinomial([[1, -1, -1]], 10)
  # samples is equivalent to tf.zeros([1, 10], dtype=tf.int64).

##### 인자:


*  <b>`logits`</b>: `[batch_size, num_classes]` shape을 갖는 2-D 텐서. 각 슬라이스 `[i, :]`는 모든 클래스에 대한 비정규화 로그 확률을 나타냅니다.
*  <b>`num_samples`</b>: 0-D.  Number of independent samples to draw for each row slice. 0-D. 각 행 슬라이스를 뽑기위한 독립적인 샘플의 갯수.
*  <b>`seed`</b>: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)를 보십시오.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `[batch_size, num_samples]` shape의 샘플들

- - -

### `tf.set_random_seed(seed)` {#set_random_seed}

그래프 수준의 난수 시드를 설정합니다.

난수 시드에 의존하는 연산들은 실제로 그래프 수준과 연산 수준의 두 가지 시드로부터 시드를 얻어냅니다. 이 연산은 그래프 수준의 시드를 설정합니다.

연산 수준의 시드와의 상호작용은 다음과 같습니다.

  1. 그래프 수준과 연산 시드가 모두 설정되어있지 않은 경우: 난수 시드는 이 연산을 위해 사용됩니다.
  2. 그래프 수준의 시드가 설정되어있고, 연산 시드는 설정되어있지 않은 경우, 시스템은  유일한 난수 시퀀스를 얻기위해 결정론적으로 그래프 수준의 시드와 함께 사용할 연산 시드를 선택합니다.
  3. 그래프 수준의 시드가 설정되어있지 않고 연산 시드만 설정되어있는 경우, 난수 시퀀스를 결정하기 위해 그래프 수준의 시드의 기본값과 지정된 연산 시드가 사용됩니다.
  4. 두 시드 모두 설정되어있을 경우, 난수 시퀀스를 결정하기 위해 두 시드가 함께 사용됩니다.

눈에 보이는 효과를 설명하기위해, 다음과 같은 예시들을 생각해봅시다:

세션간에 다른 시퀀스를 생성하기위해 그래프 수준과 연산 수준의 시드를 모두 설정하지 않습니다.

```python
a = tf.random_uniform([1])
b = tf.random_normal([1])

print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A3'
  print(sess2.run(a))  # generates 'A4'
  print(sess2.run(b))  # generates 'B3'
  print(sess2.run(b))  # generates 'B4'
```

세션간에 하나의 연산이 똑같이 반복가능한 시퀀스를 생성할 수 있도록, 연산 시드를 설정합니다.

```python
a = tf.random_uniform([1], seed=1)
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate the same
# sequence of values for 'a', but different sequences of values for 'b'.
print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A1'
  print(sess2.run(a))  # generates 'A2'
  print(sess2.run(b))  # generates 'B3'
  print(sess2.run(b))  # generates 'B4'
```

모든 연산에 의해 생성된 난수 시퀀스들이 세션간 반복이 가능하게 하기위해서, 그래프 수준의 시드를 설정합니다.

```python
tf.set_random_seed(1234)
a = tf.random_uniform([1])
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate different
# sequences of 'a' and 'b'.
print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A1'
  print(sess2.run(a))  # generates 'A2'
  print(sess2.run(b))  # generates 'B1'
  print(sess2.run(b))  # generates 'B2'
```

##### 인자:


*  <b>`seed`</b>: 정수.
