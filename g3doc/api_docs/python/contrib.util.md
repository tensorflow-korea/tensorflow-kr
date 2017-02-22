<!-- This file is machine generated: DO NOT EDIT! -->

# 유틸리티 (contrib)
[TOC]

텐서를 처리하는 유틸리티

## 여러 유틸리티 함수

- - -

### `tf.contrib.util.constant_value(tensor)` {#constant_value}

만약 효율적으로 계산이 가능하다면 tensor의 상수 값을 반환합니다.

이 함수는 텐서가 주어지면 부분적으로 평가를 진행합니다. 성공하는 경우 numpy ndarray 값을 반환합니다.

하는일(mrry): 이 함수는 손쉽게 확장이 가능하도록 gradients와 ShapeFunctions과 같은 등록 매커니즘을 고려합니다.

주의: 만약 `constant_value(tensor)` 가 non-`None` 결과를 반환하면 tensor에 별다른 값을 부여할 수 없게 됩니다. 이 함수는 구축된 그래프에 영향을 미치도록 허용합니다. 그리고 정적 형상 최적화(permits static shape optimizations)을 허용합니다. 

##### 인자:


*  <b>`tensor`</b>: 평가된 텐서

##### 반환값:

  numpy ndarray는 `tensor`의 상수 값이거나 계산하지 않았다면 None을 포함합니다.

##### 예외:


*  <b>`TypeError`</b>: 텐서가 작동하지 않는 경우에 타입에러가 발생함.


- - -

### `tf.contrib.util.make_tensor_proto(values, dtype=None, shape=None)` {#make_tensor_proto}

TensorProto 생성.

##### 인자:


*  <b>`values`</b>: Values 를 TensorProto에 둡니다.
*  <b>`dtype`</b>: Optional tensor_pb2 데이터 타입 값
*  <b>`shape`</b>: 정수 리스트 형태로 텐서의 차원을 나타냅니다.

##### 반환값:

  TensorProto는 타입에 의존적입니다. TensorProto는 "tensor_content"를 포함합니다. 파이썬 프로그램에서 직접적으로 유용한 기능은 아닙니다.
  값을 평가하기 위해 tensor_util.MakeNdarray(proto)를 이용해 proto를 다시 numpy ndarray로 변환해야합니다.

##### 예외:


*  <b>`TypeError`</b>: 타입이 제공되지 않은 경우.
*  <b>`ValueError`</b>: 인자가 부적절한 값일 경우

make_tensor_proto 는 파이썬 스칼라 값인 "values" 를 받아들입니다. values는 파이썬의 리스트 형태입니다. numpy ndarray 와 numpy scalar와 같습니다. 

만약 "values" 가 파이썬의 스칼라 혹은 리스트 형태라면, make_tensor_proto
first 는 numpy ndarray로 변환됩니다. 만약 dtype 이 없다면, numpy 데이터형이 무엇인지 추론을 시도합니다. 달리 말하면 반환되는 numpy 배열은 주어진 dtype에 호환되는 데이터타입이 됩니다.

위의 두 경우에 있어서 numpy ndarray (호출자가 제공되거나, 자동 변환이 이뤄짐)는 반드시 dtype을 참고하여 타입이 호환되도록 해야합니다.

make_tensor_proto 는 numpy array에서 tensor proto로 변환을 담당합니다.

만약 "모양"이 None 일때 결과 텐서 proto는 numpy array로 정확히 표현할 수 있습니다.

다른말로 말하면 "모양"이 텐서의 모양으로 명시되며 numpy array는 더많은 엘리먼트가 필요 없게 됩니다.

- - -

### `tf.contrib.util.make_ndarray(tensor)` {#make_ndarray}

텐서로부터 numpy ndarray 를 생성합니다.

numpy ndarray를 생성할때 텐서와 동일한 모양과 데이터가 되도록 합니다.

##### 인자:


*  <b>`tensor`</b>: TensorProto

##### 반환값:

  텐서 컨텐츠로 이뤄진 numpy 배열

##### 예외:


*  <b>`TypeError`</b>: 텐서가 타입을 지원하지 않을때 에러가 발생합니다.


- - -

### `tf.contrib.util.ops_used_by_graph_def(graph_def)` {#ops_used_by_graph_def}

그래프에 사용된 ops의 리스트를 수집합니다.

ops가 모두 등록되었다면 검증하지 않습니다.

##### 인자:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### 반환값:

  문자열 리스트를 반홥합니다. 그래프에 사용된 각 op를 네이밍합니다.

- - -

### `tf.contrib.util.stripped_op_list_for_graph(graph_def)` {#stripped_op_list_for_graph}

Collect the stripped OpDefs for ops used by a graph.
그래프에서 사용된 ops에 대해 stripped OpDefs를 수집합니다.

이 함수는 `MetaGraphDef`의 `stripped_op_list` 필드와  protos를 계산합니다. 결과는 생산자(producer)에서 소비자(consumer)로 의사소통이 이뤄집니다. 이는 C++ 함수인 
`RemoveNewDefaultAttrsFromGraphDef`를 이용해 호환성이 향상될 수 있도록 합니다.

##### 인자:

*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### 반환값:

  ops의 `OpList` 는 그래프로서 사용됩니다.

##### 예외:


*  <b>`ValueError`</b>: 만약 등록되지 않은 op가 사용된 경우


