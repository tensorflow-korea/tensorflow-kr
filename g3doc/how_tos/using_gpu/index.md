# GPU 사용하기

## 지원되는 디바이스

일반적인 시스템에는 여러 개의 계산 디바이스가 존재합니다. TensorFlow에서는 `CPU`와 `GPU` 디바이스를 지원합니다. 디바이스는 다음과 같이 `string`으로 표현됩니다.

*  `"/cpu:0"`: 시스템의 CPU를 지정함
*  `"/gpu:0"`: 시스템의 첫 번째 GPU를 지정함(있는 경우)
*  `"/gpu:1"`: 시스템의 두 번째 GPU를 지정함(세 번째 이후도 :2, :3, ... 으로 지정 가능)

만약 사용된 TensorFlow 계산 과정이 CPU와 GPU를 모두 지원한다면, 해당 계산 과정은 GPU 디바이스에 우선적으로 배치됩니다. 예로, `matmul`은 CPU와 GPU 커널 모두 가지고 있습니다. 따라서 `cpu:0`과 `gpu:0` 디바이스가 있는 시스템에서는, `gpu:0`이 `matmul`을 실행하도록 선택될 것입니다.

## 디바이스 배치 로깅

연산과 텐서가 어떤 디바이스에 배치되었는지 알아보기 위해, 세션을 만들 때 `log_device_placement` 옵션을 `True`로 설정할 수 있습니다.

```python
# 그래프를 생성합니다.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# log_device_placement을 True로 설정하여 세션을 만듭니다.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# op를 실행합니다.
print sess.run(c)
```

다음과 같은 출력 결과를 얻을 수 있습니다:

```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/gpu:0
a: /job:localhost/replica:0/task:0/gpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]

```

## 수동으로 디바이스에 배치

만약 특정 작업을 자동으로 디바이스에 배치하지 않고, 원하는 디바이스에 배치하고 싶다면, `with tf.device`를 사용할 수 있습니다. 이를 사용하여 만들어진 디바이스 컨텍스트 내에서 이루어지는 모든 연산 과정은 명시된 디바이스 내에서 일어납니다.

```python
# 그래프를 생성합니다.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# log_device_placement을 True로 설정하여 세션을 만듭니다.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# op를 실행합니다.
print sess.run(c)
```

다음과 같이 `a`와 `b`가 `cpu:0`에 배치된 것을 확인할 수 있습니다.

```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
```

## GPU 메모리 증가 허용하기

기본적으로, TensorFlow는 GPU를 쓸 때 거의 대부분의 GPU 메모리를 사용합니다. 이는 디바이스의 [메모리 단편화](https://en.wikipedia.org/wiki/Fragmentation_%28computing%29)를 방지해 메모리를 효율적으로 사용할 수 있기 때문입니다.

때때로 디바이스에서 사용 가능한 메모리의 일부분만 할당하거나, 실행 과정에서 메모리를 추가로 할당하는 것이 더 나을 수도 있습니다. TensorFlow는 이러한 과정을 위해 세션에 두 가지 설정 옵션을 제공합니다.

첫 번째는 `allow_growth` 옵션입니다. 이는 실행 과정에서 요구되는 만큼의 GPU 메모리만 할당하게 합니다. 처음에는 매우 작은 메모리만 할당합니다. 그리고 세션이 실행되면서 더 많은 GPU 메모리가 필요해지면, TensorFlow에서 필요한 메모리 영역을 증가시켜 추가로 할당하게 됩니다. 메모리 할당을 해제하면 메모리 단편화 현상이 발생할 수 있으므로 증가된 메모리는 해제되지 않습니다. 이 옵션을 사용하기 위해서, 다음과 같은 코드로 ConfigProto의 옵션을 설정할 수 있습니다.

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```

두 번째 방법은 `per_process_gpu_memory_fraction` 옵션입니다. 이는 각각의 사용 가능한 GPU에 대해 GPU 메모리의 일정 부분만 할당하게 합니다. 예로, 다음과 같은 코드로 TensorFlow에서 각각의 GPU 메모리의 40%만 할당하도록 설정할 수 있습니다.

```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```

이 설정은 TensorFlow 프로세스에 의해 사용되는 GPU 메모리의 양에 상한이 필요할 때 유용합니다.

## 멀티 GPU 시스템에서 한 개의 GPU만 사용하기

만약 시스템에 하나 이상의 GPU 디바이스가 있다면, 기본적으로는 가장 ID가 작은 GPU가 자동으로 선택됩니다. 만약 다른 GPU에서 실행하고 싶다면, 다음과 같이 디바이스를 명시해 주어야 합니다.

```python
# 그래프를 생성합니다.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# log_device_placement을 True로 설정하여 세션을 만듭니다.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# op를 실행합니다.
print sess.run(c)
```

만약 명시된 디바이스가 존재하지 않는다면, 다음과 같이 `InvalidArgumentError` 오류가 나게 됩니다.

```
InvalidArgumentError: Invalid argument: Cannot assign a device to node 'b':
Could not satisfy explicit device specification '/gpu:2'
   [[Node: b = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [3,2]
   values: 1 2 3...>, _device="/gpu:2"]()]]
```

명시된 디바이스가 존재하지 않는 경우 실행 디바이스를 TensorFlow가 자동으로 존재하는 디바이스 중 선택하게 하려면, 세션을 만들 때 `allow_soft_placement` 옵션을 `True`로 설정하면 됩니다.

```python
# 그래프를 생성합니다.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# allow_soft_placement와 log_device_placement 옵션을
# True로 설정하여 세션을 만듭니다.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# op를 실행합니다.
print sess.run(c)
```

## 여러 개의 GPU 사용하기

TensorFlow를 멀티 GPU 시스템에서 사용하는 경우, 다음의 코드와 같이 각각의 GPU에 모델을 분산시켜 구성할 수 있습니다.

```python
# 그래프를 생성합니다.
c = []
for d in ['/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# log_device_placement을 True로 설정하여 세션을 만듭니다.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# op를 실행합니다.
print sess.run(sum)
```

다음의 결과를 얻을 수 있습니다.

```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/gpu:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/gpu:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/gpu:3
Const_2: /job:localhost/replica:0/task:0/gpu:3
MatMul_1: /job:localhost/replica:0/task:0/gpu:3
Const_1: /job:localhost/replica:0/task:0/gpu:2
Const: /job:localhost/replica:0/task:0/gpu:2
MatMul: /job:localhost/replica:0/task:0/gpu:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
```

여러 개의 GPU를 사용하여 모델을 훈련하는 과정의 좋은 예시로 [cifar10 튜토리얼](../../tutorials/deep_cnn/index.md)이 있습니다.
