# 자주 하는 질문 (FAQ)

이 문서는 TensorFlow에 대해 자주 하는 몇몇 질문의 답을 제공한다.
찾는 질문이 이 곳에 없으면, 아마도 TensorFlow [커뮤니티 리소스](../resources/index.md)에서 찾을 수 있을 것이다.

[TOC]

## 특징과 호환성

#### 여러 대 컴퓨터에서 분산 트레이닝을 할 수 있는가?

할 수 있다!
TensorFlow는 버전 0.8 부터 [분산 컴퓨테이션 지원](../how_tos/distributed/index.md). TensorFlow는 하나 이상의 컴퓨터에서 멀티 디바이스(CPU와 GPU)를 지원한다. 

#### TensorFlow는 Python 3에서 동작 하는가?

0.6.0 릴리즈(2015년 12월 초)부터 Python 3.3+를 지원한다.

## TensorFlow 그래프 작성

[그래프 작성 API 문서](../api_docs/python/framework.md) 참조.

#### `c = tf.matmul(a, b)` 는 왜 매트릭스 곱 연산을 바로 실행하지 않는가??

TensorFlow 파이썬 API에서 `a`, `b`, `c` 는 [`Tensor`](../api_docs/python/framework.md#Tensor) 오브젝트다. 
`Tensor` 오브젝트는 오퍼레이션 결과에 대한 심볼릭 핸들이지만, 실제로 오퍼레이션의 결과 값을 갖고 있지는 않다.
대신, TensorFlow는 사용자가 복잡한 표현(전체 뉴럴 네트워크와 그래디언트 같은)을 데이터플로우 그래프로 작성하는 것을 권장한다. 
그렇게 해 놓으면 전체 데이터플로우 그래프(또는 이의 하위그래프)의 컴퓨테이션을 TensorFlow 
[`Session`](../api_docs/python/client.md#Session)이 실행하게 할 수 있다.
이렇게 하면 오퍼레이션을 하나씩 실행 하는 것 보다, 전체 컴퓨테이션을 보다 효율적으로 실행 할 수 있다.

#### 디바이스명은 어떻게 불리는가?

CPU 디바이스는 `"/device:CPU:0"` (또는 `"/cpu:0"`)로,
*i*th GPU 디바이스는 `"/device:GPU:i"` (또는 `"/gpu:i"`)로 지원된다.

#### 특정 디바이스에 오퍼레이션을 배치하려면 어떻게 해야 하는가?

한 디바이스에 오퍼레이션 그룹을 배치하려면 이들을 
[`with tf.device(name):`](../api_docs/python/framework.md#device)컨텍스트 
안에서 생성해라. TensorFlow에서 오퍼레이션을 디바이스에 할당하는 방법에 대한 자세한 설명 [TensorFlow를 GPU와 함께 사용하기](../how_tos/using_gpu/index.md), 
그리고 다수의 GPU를 사용하는 모델에 대한 예 [CIFAR-10 tutorial](../tutorials/deep_cnn/index.md) 
에 대한 how-to 문서를 참조해라. 

#### 가용한 텐서의 다른 타입에는 무엇이 있는가?

TensorFlow는 다양한 다른 종류의 데이터 타입과 텐서 쉐이프(shape)를 지원한다.
보다 자세한 내용은 [ranks, shapes, and types reference](../resources/dims_types.md)를 참조해라.


## TensorFlow 컴퓨테이션 실행하기

[API documentation on running graphs](../api_docs/python/client.md)를 참조해라.

#### 피딩(feeding)과 플레이스홀더(placeholder)는 어떤 관계인가?

피딩은 TensorFlow 세션 API에 있는 메커니즘으로,
실행 시간에 하나 이상의 텐서를 위해 다른 값들로 교체하는 것을 허용한다.
[`Session.run()`](../api_docs/python/client.md#Session.run)에 대한 `feed_dict` 인자는
[`Tensor`](../api_docs/python/framework.md) 오브젝트를 
numpy(또는 다른 타입) 배열로 맵(map)하는 딕셔너리(dictionary)이고,
이들은 단계(step) 실행 시에 텐서의 값으로 사용될 것이다.

입력 처럼, 항상 있어줘야 하는 텐서가 있다.
[`tf.placeholder()`](../api_docs/python/io_ops.md#placeholder) op는 
*반드시* 있어줘야 하는 텐서를 정의할 수 있도록 해주며, 옵션으로 모양(shape)의 제약(contrain) 또한 허용한다.

#### `Session.run()`과 `Tensor.eval()`의 차이는 무엇인가?

`t`가 [`Tensor`](../api_docs/python/framework.md#Tensor) 오브젝트라면,
[`t.eval()`](../api_docs/python/framework.md#Tensor.eval)은
[`sess.run(t)`](../api_docs/python/client.md#Session.run)의 속기 표현이다.
(`sess`가 현재의 [디폴트 세션](../api_docs/python/client.md#get_default_session)인 곳에서)

```python
# `Session.run()` 사용.
sess = tf.Session()
c = tf.constant(5.0)
print sess.run(c)

# `Tensor.eval()` 사용.
c = tf.constant(5.0)
with tf.Session():
  print c.eval()
```

두번째 예제에서,
세션은 [컨텍스트 관리자](https://docs.python.org/2.7/reference/compound_stmts.html#with) 처럼 동작하며,
이는 `with` 블럭 내의 디폴트 세션으로 설정하는 효과가 있다. 
컨텍스트 관리자 접근법은 간단한 유즈케이스(단위 테스트 같은)에서 보다 간결한 코드를 만들 수 있다.
다중 그래프와 세션을 다루는 코드에서는 `Sesson.run()`을 명시적으로 호출하는것이 보다 직관적일 수 있다. 

#### 세션에 라이프타임이 있는가? 중간에 있는 텐서는 어떻게 되는가?

세션은 
[variables](../api_docs/python/state_ops.md#Variable),
[queues](../api_docs/python/io_ops.md#QueueBase),
[readers](../api_docs/python/io_ops.md#ReaderBase) 같은 리소스를 가질 수 있다.
그리고 이런 리소스는 상당히 많은 메모리를 사용할 수 있다.
이런 리소스(그리고 관련된 메모리)는 세션이 닫힐 때
[`Session.close()`](../api_docs/python/client.md#Session.close) 호출에 의해
릴리즈 된다.

[`Session.run()`](../api_docs/python/client.md) 호출의 일부로 생성된 중간 과정 텐서는
호출의 마지막 단계 또는 그 이전에 해제(free)될 것이다. 

#### 런타임은 그래프의 일부분을 병렬로 실행하는가?

텐서 런타임은 다수의 다양한 차원에 대해서 그래프 실행을 병렬화 한다:

* 각 ops는 멀티코어 CPU 또는 GPU에서 멀티쓰레드를 사용하는 병렬처리 구현을 갖고 있다.

* 텐서 그래프에 있는 각 노드는 멀티 노드에서 병렬로 처리 될 수 있고,
이는 스피드업을 가능하게 만든다 [CIFAR-10 멀티 GPU를 사용한 트레이닝](../tutorials/deep_cnn/index.md).

* 세션 API는 병렬처리에서 다수의 동시적 스텝(step)(예,[Session.run()](../api_docs/python/client.md#Session.run)) 을 허용한다.
  만약 하나의 스텝이 모든 리소스를 사용하지 않는다면, 이는 런타임의 처리량 향상을 가능하게 한다. 

#### TensorFlow에서 지원하는 클라이언트 언어는?

TensorFlow는 다중 클라이언트 언어를 지원하도록 설계되었다.
현재, 가장 잘 지원 되는 클라이언트 언어는 [Python](../api_docs/python/index.md)이다.
[C++ client API](../api_docs/cc/index.md)는 그래프 론칭과 스텝을 돌리기 위한 인터페이스를 제공한다;
또한 [building graphs in C++](https://www.tensorflow.org/code/tensorflow/cc/tutorials/example_trainer.cc)라는 실험용 API도 있다. 

커뮤니티의 관심도에 따라서 더 많은 클라이언트 언어를 지원하고자 한다.
TensorFlow에는 [C-based client API](https://www.tensorflow.org/code/tensorflow/core/public/tensor_c_api.h)가 있으며,
이는 많은 다른 언어에 대한 클라이언트 제작을 쉽게 해준다.
우리는 새로운 언어로 바인딩 하는 공헌을 바라고 있다. 

#### TensorFlow는 디바이스에 있는 GPU와 CPU를 모두 사용하는가?

TensorFlow는 멀티 GPU와 CPU를 지원한다.
TensorFlow가 오퍼레이션을 디바이스에 어떻게 할당하는지에 대한 자세한 방법과
다중 GPU를 사용하는 모델의 예제를 보려면
[using GPUs with TensorFlow](../how_tos/using_gpu/index.md) 문서를 참조해라.

TensorFlow는 계산능력(Compute Capability)가 3.5 이상인 디바이스만 사용한다는 것에 주의하라.

#### reader나 queue 사용시 `Sesson.run()`은 왜 멈추는가(Hang)? 

[reader](../api_docs/python/io_ops.md#ReaderBase)와
[queue](../api_docs/python/io_ops.md#QueueBase) 클래스는
입력(또는 큐의 메모리 공간)이 가능해 질 때 까지 *block* 할 수 있는 특별한 오퍼레이션을 제공한다.
이러한 오퍼레이션들은 TensorFlow 컴퓨테이션을 다소 많이 복잡하게 하는 비용을 감수 하고서라도,
복잡한 [input pipelines](../how_tos/reading_data/index.md)을 만들 수 있도록 한다.
사용법에 대한 더 많은 정보를 원하면
[using `QueueRunner` objects to drive queues and readers](../how_tos/reading_data/index.md#creating-threads-to-prefetch-using-queuerunner-objects)
를 참조해라.

## 변수

[variables](../how_tos/variables/index.md),
[variable scopes](../how_tos/variable_scope/index.md),
[the API documentation for variables](../api_docs/python/state_ops.md)
에 있는 how-to 문서 또한 참조해라.

#### 변수의 라이프타임은 무엇인가?

변수는 세션에서
[`tf.Variable.initializer`](../api_docs/python/state_ops.md#Variable.initializer)
오퍼레이션을 처음 실행할 때 생긴다.
[`session is closed`](../api_docs/python/client.md#Session.close)
될 때 변수는 삭제된다.

#### 변수에 동시 접근 할 때 (변수는) 어떻게 되는가? 

변수는 동시에 읽고 쓰는 오퍼레이션을 허용한다.
읽어온 변수가 동시에 업데이트 되는 경우, 변수는 변경될 수 있다.
기본적으로, 변수에 대한 동시할당 오퍼레이션은 상호배제(mutual exclusion)를 사용하지 않아도 허용된다.
변수에 값 할당 시 락(lock)을 얻어 오려면 `use_lock=True`를
[`Variable.assign()`](../api_docs/python/state_ops.md#Variable.assign)에 전달하라.

## 텐서 쉐이프(shape)

[`TensorShape` API documentation](../api_docs/python/framework.md#TensorShape)
도 참조하라.

#### 파이썬 환경에서 텐서 쉐이프를 어떻게 결정 할 수 있는가?

TensorFlow에서, 텐서는 정적 (inferred) 쉐이프와 동적 (true) 쉐이프를 갖고 있다. 정적 쉐이프는  
[`tf.Tensor.get_shape()`](../api_docs/python/framework.md#Tensor.get_shape)
메소드로 읽을 수 있다. 이 쉐이프는 텐서를 사용하는데 사용된 오퍼레이션으로 부터 추론할 수 있다.
그리고 [partially complete](../api_docs/python/framework.md#TensorShape)일 것이다.
정적 쉐이프가 완전히 정의되지 않았다면, 동적 쉐이프 `Tensor` `t`가 
[`tf.shape(t)`](../api_docs/python/array_ops.md#shape)
의 평가에 의해 결정 될 수 있다.

#### `x.set_share()`와 `x = tf.reshape(x)`간의 차이는 무엇인가?

[`tf.Tensor.set_shape()`](../api_docs/python/framework.md)
메소드는 `Tensor` 오브젝트의 정적 쉐이프를 업데이트 한다.
그리고 전형적으로 쉐이프를 직접 추론할 수 없을 때 추가 쉐이프 정보를 제공하기 위해 사용한다.
이는 텐서의 동적 쉐이프를 변경하지 않는다. 

[`tf.reshape()`](../api_docs/python/array_ops.md#reshape)
오퍼레이션은 다른 동적 쉐이프의 새로운 텐서를 생성한다.

#### 변수 배치 크기로 동작하는 그래프는 어떻게 만들어야 하는가?

변수 배치 크기로 동작하는 그래프를 만들면 유용한 경우가 많다.
예를 들면, 같은 코드로 (미니-)배치 트레이닝과 싱글-인스턴스 추론을 할 수 있다. 결과 그래프는 
[saved as a protocol buffer](../api_docs/python/framework.md#Graph.as_graph_def)와
[imported into another program](../api_docs/python/framework.md#import_graph_def)이 될 수 있다.

변수 크기 그래프를 만들 때, 가장 중요한 것은 파이썬 상수로 배치 크기를 인코드 하면 안되고,
대신 심볼릭 `Tensor` 를 사용해야 한다. 아래에 유용한 팁이 있다.

* `input`으로 불리는 `Tensor`로 부터 배치 차원을 뽑아내기 위해
[`batch_size = tf.shape(input)[0]`](../api_docs/python/array_ops.md#shape)
을 사용해라. 그리고 `batch_size`로 불리는 `Tensor`에 저장해라.

* `tf.reduce_sum(...) / batch_size` 대신
  [`tf.reduce_mean()`](../api_docs/python/math_ops.md#reduce_mean)를 사용해라.

* [placeholders for feeding input](../how_tos/reading_data/index.md#feeding)를 사용 한다면 
  [`tf.placeholder(..., shape=[None, ...])`](../api_docs/python/io_ops.md#placeholder)
  로 플레이스홀더(placeholder)를 생성해서 변수 배치 차원을 명시 할 수 있다.
  쉐이프의 `None` 엘리먼트는 변수 크기 차원에 대응된다.

## TensorBoard

#### TensorFlow 그래프를 어떻게 가시화 할 수 있는가?

[graph visualization tutorial](../how_tos/graph_viz/index.md)를 참조해라.

#### TensorBoard에 데이터를 보내는 가장 간단한 방법은 무엇인가?

TensorFlow 그래프에 요약 ops를 추가하고, 요약 내용을 로그 디렉토리에 쓰기 위해
[`SummaryWriter`](../api_docs/python/train.md#SummaryWriter)를 사용해라.
그리고 나서, TensorBoard를 다음 명령으로 시작해라.

    python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory

보다 자세한 내용은 
[Summaries and TensorBoard tutorial](../how_tos/summaries_and_tensorboard/index.md)
를 참조해라.

#### TensorBoard를 띄울 때 마다, 네트워크 보안 팝업이 뜬다.

--host=localhost 옵션을 줘서, '0.0.0.0'이 아닌 localhost로 서비스 되도록 변경할 수 있다.
이렇게 하면 보안 경고가 안 나올 것이다.

## TensorFlow 확장

how-to 문서인
[adding a new operation to TensorFlow](../how_tos/adding_an_op/index.md)
를 참조해라.

#### 내가 가진 데이터는 커스텀 포맷이다. TensorFlow로 읽으려면 어떻게 해야 하는가?

커스텀 포맷 데이터를 처리하는 옵션이 두 가지 있다.

쉬운 옵션은 파이썬으로 파싱 코드를 작성해서 데이터를 numpy 배열로 변경하고
[`tf.placeholder()`](../api_docs/python/io_ops.md#placeholder)
에 그 데이터 텐서를 넣는것이다.

보다 자세한 내용은
[using placeholders for input](../how_tos/reading_data/index.md#feeding)를 봐라.
이 접근법은 빨리 만들어서 돌려보기 좋지만, 파싱이 성능 병목이 될 수 있다.

보다 효율적인 옵션은
[add a new op written in C++](../how_tos/adding_an_op/index.md)
를 이용해서 당신의 데이터 포맷을 파싱하는 op를 추가하는 것이다.
[guide to handling new data formats](../how_tos/new_data_formats/index.md)에
이를 처리하는 절차에 대한 더 많은 정보가 있다.

#### 입력의 개수가 변경되는 것을 처리하는 오퍼레이션을 어떻게 정의해야 하는가?

TensorFlow op 등록 메커니즘은 입력 정의를 허용한다.
그런 입력에는 싱글 텐서, 동일 타입의 텐서 리스트 (예를 들면, 가변 길이 목록의 텐서를 같이 등록할 때),
또는 이종 타입의 텐서 리스트 (예를 들면, 큐에 텐서 튜플을 넣을 때).
이종 입력 타입을 정의하는 방법에 대한 보다 자세한 내용은
[adding an op with a list of inputs or outputs](../how_tos/adding_an_op/index.md#list-inputs-and-outputs)
를 참조해라.

## Miscellaneous

## 기타사항

#### TensorFlow의 코딩 스타일 컨벤션은 무엇인가?

TensorFlow 파이썬 API는 
[PEP8](https://www.python.org/dev/peps/pep-0008/) 컨벤션을 따른다. 
<sup>*</sup> 특히, 클래스는 `CamelCase`, 함수와 속성은 `snake_case`를 사용한다.
또한, [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
도 따른다. 

TensorFlow C++ 코드 베이스는 
[Google C++ style guide](http://google.github.io/styleguide/cppguide.html)
를 따른다.

(<sup>*</sup> 하나의 예외사항: 4-space 들여쓰기 대신 2-space 들여쓰기를 사용한다.)

