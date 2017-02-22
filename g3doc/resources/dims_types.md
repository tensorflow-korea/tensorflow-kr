# Tensor Ranks, Shapes, Type

TensorFlow 프로그램은 모든 데이터를 tensor 데이터 구조를 사용해서 표현한다.
TensorFlow의 tensor는 n-차원 배열 또는 리스트라고 생각해도 된다.
하나의 tensor는 정적 타입과 동적 차원을 갖고 있다.
컴퓨테이션 그래프의 노드들은 오직 tensor만을 전달 할 수 있다.

## Rank

TensorFlow 시스템에서, tensor는 *rank*라는 차원 단위로 표현된다.
Tensor rank는 행렬의 rank와 다르다.
Tensor rank(*order*, *degree*, *-n_dimension* 으로도 언급됨)는 tensor의 차원수다.
예를 들어, 아래 tensor(Python 리스트로 정의)의 rank는 2다.

    t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

rank 2인 tensor는 행렬, rank 1인 tensor는 벡터로 생각할 수 있다. 
rank 2인 tensor는 `t[i, j]` 형식으로 원소에 접근할 수 있다.
rank 3인 tensor는 `t[i, j, k]` 형식으로 원소를 지정할 수 있다.

Rank | Math entity | Python example
--- | --- | ---
0 | Scalar (magnitude only) | `s = 483`
1 | Vector (magnitude and direction) | `v = [1.1, 2.2, 3.3]`
2 | Matrix (table of numbers) | `m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`
3 | 3-Tensor (cube of numbers) | `t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]`
n | n-Tensor (you get the idea) | `....`

## Shape

TensorFlow 문서는 tensor 차원을 표현할 때 세 가지 기호를 사용한다. rank, shape, 차원수.
아래 표는 그 세 가지의 관계를 보여준다:

Rank | Shape | Dimension number | Example
--- | --- | --- | ---
0 | [] | 0-D | A 0-D tensor.  A scalar.
1 | [D0] | 1-D | A 1-D tensor with shape [5].
2 | [D0, D1] | 2-D | A 2-D tensor with shape [3, 4].
3 | [D0, D1, D2] | 3-D | A 3-D tensor with shape [1, 4, 3].
n | [D0, D1, ... Dn-1] | n-D | A tensor with shape [D0, D1, ... Dn-1].

Shape는 Python 리스트 / 정수형 튜플 또는
[`TensorShape` class](../api_docs/python/framework.md#TensorShape)로 표현 할 수 있다. 

## Data types

Tensor는 차원 말고도 데이터 타입도 갖는다.
아래의 데이터 타입을 tensor에 지정할 수 있다.

Data type | Python type | Description
--- | --- | ---
`DT_FLOAT` | `tf.float32` | 32 비트 부동 소수.
`DT_DOUBLE` | `tf.float64` | 64 비트 부동 소수.
`DT_INT8` | `tf.int8` | 8 비트 부호 있는 정수.
`DT_INT16` | `tf.int16` | 16 비트 부호 있는 정수.
`DT_INT32` | `tf.int32` | 32 비트 부호 있는 정수.
`DT_INT64` | `tf.int64` | 64 비트 부호 있는 정수.
`DT_UINT8` | `tf.uint8` | 8 비트 부호 없는 정수.
`DT_STRING` | `tf.string` | 가변 길이 바이트 배열. Tensor의 각 원소는 바이트 배열.
`DT_BOOL` | `tf.bool` | 불리언.
`DT_COMPLEX64` | `tf.complex64` | 2개의 32 비트 부동 소수로 만든 복소수 : 실수부 + 허수부
`DT_COMPLEX128` | `tf.complex128` | 2개의 64 비트 부동 소수로 만든 복소수 : 실수부 + 허수부
`DT_QINT8` | `tf.qint8` | 8 비트 부호 있는 정수로 quantized Ops에서 사용.
`DT_QINT32` | `tf.qint32` | 32 비트 부호 있는 정수로 quantized Ops에서 사용.
`DT_QUINT8` | `tf.quint8` | 8 비트 부호 없는 정수로 quantized Ops에서 사용.


(역주: `quantized op`는 fixed-point 데이터로써, 
[quantized op](https://github.com/tensorflow/tensorflow/issues/15)에 의하면 아직 문서화 되지 않은 기능.)
