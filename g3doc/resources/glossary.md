# 어휘 (Glossary)

**Broadcasting operation**
****

텐서 인자의 구조(shape)와의 호환을 위해 [numpy-style broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)을 사용하는 연산입니다.

**Device**

계산을 실행할 수 있으며 자체 주소 공간을 갖는 GPU나 CPU와 같은 하드웨어의 일부입니다.

**eval**

값을 결정하는데 필요한 그래프 계산을 트리거링하며 `Tensor`의 값을 반환하는 `Tensor`의 메서드입니다. 세션에서 시작된 그래프에서 `Tensor`의 `eval()`을 호출하기만 하면됩니다.

**Feed**

세션에서 시작된 그래프에 있는 노드로 직접 텐서를 패치하는 TensorFlow의 메커니즘입니다. 그래프를 생성하는 때가 아닌 그래프의 실행을 트리거링할 때 피드를 적용합니다. 피드는 임시로 노드를 텐서값으로 바꿉니다. 계산을 시작하는 `run()` 또는 `eval()` 호출에 피드 데이터를 인자로써 공급합니다. 실행 후에는 피드가 사라지며 원래 노드의 정의가 남습니다. 일반적으로 그것들을 생성하기위해 `tf.placeholder()`를 사용하여 "feed" 노드가 될 특정한 노드들을 지정합니다. 좀 더 자세한 내용은 [Basic Usage](../get_started/basic_usage.md)를 보십시오.

**Fetch**

세션에서 시작된 그래프에서 텐서를 검색하기위한 TensorFlow의 메커니즘입니다. 그래프를 생성하는 때가 아닌 그래프의 실행을 트리거링할 때 페치를 검색합니다. 노드 또는 노드들의 텐서값을 가져오기위해 `Session` 객체에서 `run()`을 호출하여 그래프를 실행시키고 검색할 노드명 리스트를 전달합니다. 좀 더 자세한 내용은 [Basic Usage](../get_started/basic_usage.md)를 보십시오.

**Graph**

방향성 비순환 그래프로 계산을 나타냅니다. 그래프의 노드들은 수행되어야 하는 연산들을 나타냅니다. 그래프의 엣지는 데이터 또는 제어 종속성을 나타냅니다. `GraphDef`는 그래프를 시스템 (API)에 서술하기위해 사용되는 프로토콜 버퍼이며 `NodeDef`(아래를 보십시오.)의 컬렉션으로 이루어져 있습니다. `GraphDef`는 조작하기 쉬운 (C++) `Graph` 객체로 변환될 수 있습니다.

**IndexedSlices**

파이썬 API에서, 텐서의 첫번째 차원만을 따르는 희소 텐서를 나타내는 TensorFlow의 표현식입니다. 만약 텐서가 `k` 차원이면, `IndexedSlices` 인스턴스는 논리적으로 텐서의 첫번째 차원을 따르는 `(k-1)` 차원 슬라이스들의 컬렉션을 나타냅니다. 슬라이스의 인덱스는 1차원 벡터와 합쳐진 상태로 저장이 되고, 각 슬라이스는 하나의 `k` 차원 텐서의 형태로 합쳐집니다. 만약 첫번째 차원에서 희소성이 제한되지 않는다면  `SparseTensor`를 사용하십시오.

**Node**

그래프의 요소.

연산을 설정하는데 필요한 `attrs`의 값들을 포함해 특정한 계산 `Graph`에서 하나의 노드로써 특정한 연산을 어떻게 실행시키는지에 대한 방법을 서술합니다. 다형적 연산을 위해 `attrs`는 `Node`의 시그니쳐를 완전히 결정지을 수 있는 충분한 정보를 포함합니다. 자세한 건 `graph.proto`를 보십시오.

**Op (operation)**

TensorFlow 런타임에서: `add`나 `matmul` 또는 `concat`과 같은 연산의 타입입니다. [how to add an
op](../how_tos/adding_an_op/index.md)에 설명된대로 런타임에 새로운 연산을 추가할 수 있습니다.

파이썬 API에서: 그래프의 노드입니다. 연산은 [`tf.Operation`](../api_docs/python/framework.md#Operation) 클래스의 인스턴스로 나타냅니다. `Operation`의 `type` 프로퍼티는 `add`나 `matmul`과 같은 노드에 대한 실행 연산을 나타냅니다.

**Run**

시작된 그래프에서 연산 실행의 액션입니다. 그래프가 `Session`에서 시작되어야 합니다.

파이썬 API에서: `Session` 클래스 [`tf.Session.run`](../api_docs/python/client.md#Session)의 메서드입니다. 피드와 페치를 하기 위한 텐서를 `run()` 호출에 전달할 수 있습니다.

C++ API에서: [`tensorflow::Session`](../api_docs/cc/ClassSession.md)의 메서드입니다.

**Session**

시작된 그래프를 나타내는 런타임 객체입니다. 그래프에서 연산을 실행하기 위한 메서드들을 제공합니다.

파이썬 API에서: [`tf.Session`](../api_docs/python/client.md#Session)

C++ API에서: 그래프를 시작하고 연산을 실행할 때 사용되는 클래스 [`tensorflow::Session`](../api_docs/cc/ClassSession.md)

**Shape**

텐서의 차원과 크기.

시작된 그래프에서: 노드 사이를 흐르는 텐서의 프로퍼티. 몇몇 연산은 그들의 입력의 구조(shape)에 대한 엄격한 요구조건을 가지고 있으며, 안맞을시 런타임에서 에러를 리포팅합니다.

파이썬 API에서: 그래프 생성 API에 있는 파이썬 `Tensor`의 속성. 생성 도중 텐서의 구조(shape)는 부분적으로만 알 수 있거나 아예 모를수도 있습니다. [`tf.TensorShape`](../api_docs/python/framework.md#TensorShape)를 보십시오.

C++ API에서: 텐서의 구조(shape)을 나타내는데 사용되는 클래스 [`tensorflow::TensorShape`](../api_docs/cc/ClassTensorShape.md)

**SparseTensor**

파이썬 API에서, 임의의 위치에서 드물게 존재하는 텐서를 나타내는 TensorFlow의 표현식입니다. `SparseTensor`는 딕셔너리-키 포맷을 사용해 값의 인덱스에 따라 비어 있지 않는 값들만을 저장합니다. 다시 말하면, `m`개의 비어 있지 않는 값들이 있을때, 이는 길이가 `m`인 값의 벡터와 m개의 인덱스의 행을 갖는 행렬을 유지합니다. 효율성을 위해, `SparseTensor`는 차원수가 증가함에 따라 정렬된 (즉, 행 위주 순서) 인덱스가 필요합니다. 만약 희소성이 첫번째 차원만 따르는 경우엔 `IndexedSlices`를 사용하십시오.

**Tensor**

`Tensor`는 타입을 가진 다차원 배열입니다. 예를 들면, `[batch, height, width, channel]` 차원을 갖는 이미지의 작은 배치를 나타내는 실수형 숫자들의 4차원 배열이 있습니다.

시작된 그래프에서: 노드 사이를 흐르는 데이터의 타입.

파이썬 API에서: 그래프에 추가된 연산의 출력과 입력을 나타내는데 사용되는 클래스[`tf.Tensor`](../api_docs/python/framework.md#Tensor). 이 클래스의 인스턴스는 데이터를 저장하지 않습니다.

C++ API에서: [`tensorflow::Tensor`](../api_docs/cc/ClassTensor.md)를 호출하는 [`Session::Run()`](../api_docs/cc/ClassSession.md)에서 반환되는 텐서를 나타내는데 사용되는 클래스. 이 클래스의 인스턴스는 데이터를 가지고 있습니다.
