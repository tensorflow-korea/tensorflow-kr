<!-- This file is machine generated: DO NOT EDIT! -->

# 파이썬 함수 래핑하기

참고 : `Tensor`를 인자로 받는 함수들은 [`tf.convert_to_tensor`](framework.md#convert_to_tensor)의 인자로 들어갈 수 있는 값들 또한 받을 수 있습니다.

[TOC]

## 스크립트 언어 연산자

TensorFlow는 python/numpy 함수들을 TensorFlow의 연산자로써 래핑할 수 있도록 해줍니다.

## 다른 함수와 클래스들
- - -

### `tf.py_func(func, inp, Tout, name=None)` {#py_func}

python 함수를 래핑하고 이를 tensorflow의 연산자로써 사용합니다.

`func`로 주어지는 python 함수는 numpy 배열을 입력으로 받고 numpy 배열을 출력합니다. 예를 들면,

```python
def my_func(x):
  # x는 아래의 placeholder의 값을 가지는 numpy 배열이 될 것입니다.
  return np.sinh(x)
inp = tf.placeholder(tf.float32, [...])
y = py_func(my_func, [inp], [tf.float32])
```

위의 스니펫은  그래프의 연산으로 numpy의 sinh(x)를 호출하는 tf 그래프를 구성합니다.

##### 인자:


*  <b>`func`</b>: python 함수.
*  <b>`inp`</b>: `Tensor`의 리스트.
*  <b>`Tout`</b>: `func`의 반환값을 나타내는 tensorflow 데이터 타입의 리스트.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `func`를 통해 계산된 `Tensor`의 리스트.
