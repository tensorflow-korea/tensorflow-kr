# 툴 개발자의 TensorFlow 모델 파일에 대한 가이드

대부분의 사용자는 TensorFlow가 디스크에 데이터를 어떻게 저장하는지에 대한 내부적인 세부 사항들에 대해 신경쓸 필요가 없지만, 만약 당신이 툴 개발자라면 그럴 필요가 있습니다. 예를 들면, 당신은 모델을 분석하고 싶다거나 TensorFlow와 다른 포맷 사이에서 상호 변환을 하고싶을 수 있습니다. 이 가이드는 각종 툴들을 좀 더 쉽게 개발할 수 있도록 모델 데이터를 가진 메인 파일들을 어떻게 다룰 수 있는지에 대한 상세 내용들을 설명합니다.

<!--[TOC]-->

## 프로토콜 버퍼 (Protocol Buffers)

TensorFlow의 모든 파일 포맷은 [Protocol Buffers](https://developers.google.com/protocol-buffers/?hl=en)에 기반하므로, 그들이 작동하는 방법에 대해 알아가는 것이 좋습니다. 요약하면 텍스트 파일에 데이터 구조를 정의하면, 프로토콜 버퍼 툴이 C, Python, 그리고 익숙한 방법으로 데이터를 로드, 저장 그리고 접근할 수 있는 다른 언어로 클래스를 생성합니다. 우리는 종종 프로토콜 버퍼를 protobufs라 부르며, 이 가이드에서는 이 컨벤션을 사용할 것입니다.

## GraphDef

TensorFlow에서 계산의 기본은 `Graph` 객체입니다. 이는 각각이 연산을 나타내고, 입력과 출력으로써 서로 연결되어 있는 노드들의 네트워크를 가지고 있습니다. `Graph` 객체를 생성한 후엔 `GraphDef` 객체를 반환하는 `as_graph_def()`를 호출함으로써 이를 저장할 수 있습니다.

GraphDef 클래스는 [tensorflow/core/framework/graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)에 정의된 프로토콜 버퍼 라이브러리에 의해 생성되는 객체입니다. 프로토콜 버퍼 툴은 이 텍스트 파일을 파싱하고 그래프 정의를 로딩, 저장 및 조작하는 코드를 생성합니다. 모델을 나타내는 단일 TensorFlow 파일이 있다면, 이는 프로토콜 버퍼 코드에 의해 저장된 `GraphDef` 객체들중 하나를 직렬화한 버전을 포함할 가능성이 높습니다.

이 생성된 코드는 디스크로부터 GraphDef 파일들을 저장하고 로드하는데 쓰입니다. 이를 좀 더 깊게 살펴볼 수 있는 좋은 예시는 [graph_metrics.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/graph_metrics.py)입니다. 이 파이썬 스크립트는 저장된 그래프 정의를 가지고 성능 및 리소스 통계를 추정하기 위해 모델을 분석합니다. 실제로 모델을 로드하는 코드는 다음과 같을 것입니다: 

```python
graph_def = graph_pb2.GraphDef()
```

이 라인은 graph.proto에 텍스트로 정의되며 생성된 `GraphDef` 클래스의 빈 객체를 생성합니다. 이는 우리가 가진 파일의 데이터를 채울 객체입니다. 

```python
with open(FLAGS.graph, "rb") as f:
```

여기서 스크립트에 전달한 경로에 대한 파일 핸들을 얻습니다.

```python
  if FLAGS.input_binary:
    graph_def.ParseFromString(f.read)
  else:
    text_format.Merge(f.read(), graph_def)
```

## 텍스트 또는 바이너리?

프로토콜 버퍼가 저장할 수 있는 포맷은 두 가지 있습니다. 텍스트 포맷 (Text Format)은 사람이 읽을 수 있는 형태이며, 디버깅과 편집이 편리하지만, 가중치와 같은 수치 데이터가 저장될 경우 커질 수 있습니다. 이에 대한 간단한 예제는 [graph_run_run2.pbtxt](https://github.com/tensorflow/tensorflow/blob/ae3c8479f88da1cd5636b974f653f27755cb0034/tensorflow/tensorboard/components/tf-tensorboard/test/data/graph_run_run2.pbtxt)에서 볼 수 있습니다.

바이너리 포맷 (Binary Format) 파일은 비록 읽기는 어렵지만, 같은 내용의 텍스트 포맷보다 훨씬 작은 크기를 갖습니다. 우리는 적절한 함수를 호출할 수 있도록 사용자에게 입력 파일이 바이너리인지 텍스트인지 구분할 수 있는 플래그를 제공하도록 요청합니다. 큰 바이너리 파일에 대한 예제는 [inception_dec_2015.ziparchive](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip)의 `tensorflow_inception_graph.pb`에서 볼 수 있습니다.

API 자체는 조금 혼란스러울 수 있습니다 - 텍스트 파일을 로드 할 때에는 `text_format` 모듈에 있는 유틸리티 함수를 사용하는 반면, 바이너리 호출은 실제로 `ParseFromString()`를 사용합니다.

## 노드 (Nodes)

`graph_def` 변수로 파일을 로드했다면, 이제 파일 데이터에 접근할 수 있습니다. 대부분의 실용적인 목적을 위한, 중요한 부분은 노드의 리스트를 노드 멤버에 저장하는 것입니다. 여기에 노드를 순회하는 코드가 있습니다.  

```python
for node in graph_def.node
```
각 노드는 `NodeDef` 객체이며, 이 또한 graph.proto에 정의되어 있습니다. 이들은 TensorFlow 그래프의 기본적인 빌딩 블록이고 각각은 입력 커넥션과 함께 하나의 연산을 정의합니다. 아래에 `NodeDef`의 멤버들이 있으며, 그들의 의미도 설명합니다.

### `name`

모든 노드는 그래프의 다른 어떤 노드들에 의해서도 사용되지 않는 유일한 식별자를 가져야 합니다. 만약 파이썬 API을 사용해 그래프를 생성하면서 "MatMul"과 같이 연산의 명칭과 "5"와 같이 단조 증가 숫자와 연결하는등의 식별자를 지정하지 않으면, 이는 대신 선택해줄 것입니다. 임의의 식별자를 대신 선택해줄 것입니다. 명칭은 노드끼리의 연결을 정의할 때와 그래프를 실행할 때 전체 그래프를 위한 입력과 출력을 설정할 때에 사용됩니다.

### `op`

이는 실행될 연산을 정의하는데, 예를 들면 `Add`, `MatMul`, 또는 `"Conv2D"`가 있습니다. 그래프가 실행될 때, 이 연산 명칭은 구현을 찾기위해 레지스트리에서 조회됩니다. 레지스트리는 [tensorflow/core/ops/nn_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc)에 있는것들처럼, `REGISTER_OP()` 매크로를 호출함으로써 채워집니다.

### `input`

문자열의 리스트, 문자열들은 각각 다른 노드의 명칭이며, 선택적으로 콜론(colo, ':')과 출력 포트수가 따라붙습니다. 예를 들면, 두 개의 입력을 갖는 노드는 `["some_node_name", "another_node_name"]` (이는 `["some_node_name:0", "another_node_name:0"]`와 동일합니다.) 형태의 리스트를 가질 것이며, 노드의 첫번째 입력을 `"some_node_name"`의 명칭을 갖는 노드의 첫번째 출력으로, 그리고 `"another_node_name"`의 첫번째 출력으로 두번째 입력을 정의할 것입니다.

### `device`

이는 분산된 환경 또는 강제로 연산을 CPU나 GPU 위에 놓고 싶을 때 노드가 어디에서 실행될 것인지를 정의하기때문에, 대부분의 상황에선이를 무시할 수 있습니다.

### `attr`

이는 노드의 모든 속성들을 가지고 있는 key/value 저장소입니다. 이들은 컨볼루션에 대한 필터 사이즈나 상수 연산의 값처럼 런타임때 변경할 수 없는 노드의 영속적인 프로퍼티입니다. 문자열부터, 정수, 텐서값의 배열까지 속성값의 타입들이 매우 많을 수 있기 때문에, [tensorflow/core/framework/attr_value.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)에 그것들을 가지고 있는 데이터 구조를 정의하는 별도의 프로토콜 버퍼 파일이 있습니다.

각 속성은 유일한 명칭을 가지고 있으며, 예상되는 속성들은 연산이 정의될 때 리스팅됩니다. 만약 한 속성이 노드엔 존재하지 않지만 연산 정의에서 기본값을 가지고 있을 때, 그래프 생성시 그 기본값이 사용됩니다.

파이썬에서 `node.name`, `node.op` 등을 호출함으로써 이 모든 멤버들에 접근할 수 있습니다. `GraphDef`에 저장된 노드의 리스트는 모델 아키텍처의 전체 정의입니다.

## 프리징 (Freezing)

이에 대한 하나 혼란스러운 부분은 가중치는 보통 트레이닝중 파일 포맷에 저장되지 않는다는 것입니다. 대신에, 그들은 분리된 체크포인트 파일에서 유지되며, 그래프에는 가중치들이 초기화될 때 최신값을 로드하는 `Variable` 연산이 있습니다. 종종 프로덕션에 배포할 때 나뉘어진 파일들을 갖고 있다는건 매우 불편하기 때문에, 그래프 정의와 체크포인트셋을 받아 하나의 파일로 묶어주는 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) 스크립트가 있습니다.

프리징이 하는 일은 `GraphDef`을 로드하고, 최신 체크포인트 파일로부터 모든 변수들의 값을 받아온 후, 각 `Variable` 연산을 각 속성에 저장된 가중치들의 수치 데이터를 가진 `Const`로 교체합니다. 그 다음에 이는 정방향 추론에 쓰이지 않는 관계 없는 노드들은 모두 제거하고, 하나의 출력 파일에 결과 `GraphDef`를 저장합니다.

## 가중치 포맷 (Weight Formats)

만약 신경망에 대한 TensorFlow 모델을 다루고 있다면, 가장 일반적인 문제중 하나는 가중치값을 추출 및 해석하는 것입니다. 그들을 저장하는 일반적인 방법은, 예시로 freeze_graph 스크립트를 통해 생성된 그래프에서, 가중치 `Tensors`를 포함하고 있는 `Const` 연산으로 저장하는 것입니다. 이들은 [tensorflow/core/framework/tensor.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)에 정의되어 있고, 값 뿐만 아니라 데이터의 사이즈와 타입에 대한 정보도 포함하고 있습니다. 파이썬에서 `some_node_def.attr['value'].tensor`와 같은 것을 호출함으로써 `Const` 연산을 나타내는 `NodeDef`로부터 `TensorProto` 객체를 얻을 수 있습니다.

이는 가중치 데이터를 나타내는 객체를 줄 것입니다. 데이터 자체는 객체의 타입에 의해 가리켜지는 형태로 _val 접미어와 함께 리스트의 하나로 저장될 것입니다, 예로 32 비트 실수 데이터 타입은  `float_val`입니다.

서로 다른 프레임워크간 변환 작업시 컨볼루션 가중치 값의 순서는 종종 다루기가 까다롭습니다. TensorFlow에서, `Conv2D`연산을 위한 필터 가중치들은 두번째 입력에 저장되며, `[filter_height, filter_width, input_depth, output_depth]`의 순서가 될 것으로 예상됩니다. 여기서 1씩 증가하는 filter_count는 메모리에서 인접한 값으로 이동함을 의미합니다.

이 개요가 TensorFlow 모델 파일에서 무슨 일들이 일어나고 있는지에 대한 더 나은 아이디어를 제공하길 바라며, 당신이 그들을 다루어야할 경우 도움이 될 것입니다.
