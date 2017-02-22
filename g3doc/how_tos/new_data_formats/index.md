# 사용자 지정 데이터 리더

전제 조건:

*   C++에 대한 약간의 익숙함
*   [다운로드 받은 TensorFlow 소스](../../get_started/os_setup.md#installing-from-sources)가 있어야 하고, 
    빌드 할 수 있어야 합니다.

파일에 지원하는 작업은 두 가지 형식으로 나뉩니다:

*   파일 형식: 파일에서 *레코드*(모든 문자열이 될 수 있음)를 읽는데 *리더(Reader)* 오퍼레이션(Op)을 사용합니다.
*   레코드 형식 : TensorFlow에서 사용할 수 있는 텐서(tensors)로 문자열 레코드를 설정하는데 디코더 또는 구문 분석 오퍼레이션(Ops)을 사용합니다.

예를 들어,
[CSV 파일](https://en.wikipedia.org/wiki/Comma-separated_values)을 읽는데,
[텍스트 파일에 대한 리더](../../api_docs/python/io_ops.md#TextLineReader)
이후에 
[텍스트 라인으로부터 CSV 데이터를 분석하는 오퍼레이션(Op)](../../api_docs/python/io_ops.md#decode_csv)을 사용합니다.

[TOC]

## 파일 형식의 리더 작성

`리더`는 파일에서 레코드를 읽는 것입니다. 리더 오퍼레이션(Ops)에 대한 몇 가지 예제는
이미 TensorFlow에 내장되어 있습니다.

*   [`tf.TFRecordReader`](../../api_docs/python/io_ops.md#TFRecordReader)
    ([`kernels/tf_record_reader_op.cc` 소스](https://www.tensorflow.org/code/tensorflow/core/kernels/tf_record_reader_op.cc))
*   [`tf.FixedLengthRecordReader`](../../api_docs/python/io_ops.md#FixedLengthRecordReader)
    ([`kernels/fixed_length_record_reader_op.cc` 소스](https://www.tensorflow.org/code/tensorflow/core/kernels/fixed_length_record_reader_op.cc))
*   [`tf.TextLineReader`](../../api_docs/python/io_ops.md#TextLineReader)
    ([`kernels/text_line_reader_op.cc` 소스](https://www.tensorflow.org/code/tensorflow/core/kernels/text_line_reader_op.cc))

여러분은 이것들이 모두 같은 인터페이스를 노출하는 것을 확인할 수 있으며, 유일한 차이점은 
생성자에 있습니다. 가장 중요한 메소드는 `read`입니다. 
`read`는 큐를 인자로 사용해서 필요할 때마다 언제든지 읽을 파일 이름을 알 수 있습니다.
(e.g. `read` 오퍼레이션(op)이 처음 실행되거나 이전 `read`가 파일에서 마지막 레코드를 읽을 때).
그것은 두 개의 스칼라 텐서(tensors)를 생성합니다: 문자열 키와 문자열 값.

`SomeReader`라는 새로운 리더를 생성하려면 다음을 수행해야 합니다 :

1.  C++ 에서,
    [`tensorflow::ReaderBase`](https://www.tensorflow.org/code/tensorflow/core/kernels/reader_base.h)에 `SomeReader`라는 서브클래스 정의.
2.  C++ 에서, `"SomeReader"`라는 이름으로 새로운 리더 오퍼레이션(op)과 커널 등록.
3.  Python 에서, [`tf.ReaderBase`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py)에 `SomeReader`라는 서브클래스 정의.

여러분은 모든 C++코드를 `tensorflow/core/user_ops/some_reader_op.cc` 파일에 집어넣을 수 있습니다.
파일을 읽는 코드는 C++ `ReaderBase`의 서브 클래스에서 실행되고, 
[`tensorflow/core/kernels/reader_base.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/reader_base.h)에 정의되어 있습니다.
여러분은 다음 메소드를 구현해야 합니다. 

*   `OnWorkStartedLocked`: 다음 파일 열기
*   `ReadLocked`: 레코드를 읽거나 EOF/error 리포트
*   `OnWorkFinishedLocked`: 현재 파일을 닫고
*   `ResetLocked`: 예를들어 에러가 발생한 후에 깨끗하게 만듦

`ReaderBase` 내에 "Locked"으로 끝나는 이름을 갖는 메소드는 호출되기 전에 확실하게 뮤텍스(mutex)를 습득하므로,
여러분은 일반적으로 쓰레드 세이프(thread safety)를 걱정할 필요가 없습니다. 
(클래스의 멤버로만 보호됩니다, 전역으로 보호되는 것이 아닙니다).

`OnWorkStartedLocked`의 경우, 열려 있는 파일의 이름은 `current_work()` 메소드에서 반환된 값입니다.
`ReadLocked`은 이런 시그니처를 갖습니다.

```c++
Status ReadLocked(string* key, string* value, bool* produced, bool* at_end)
```

`ReadLocked`이 파일의 레코드를 성공적으로 읽으면, 아래 값을 채웁니다:

*   `*key`: 레코드의 식별자로, 사용자는 이 레코드를 다시 찾을때 사용할 수 있습니다. 
여러분은 `current_work()`에 파일 이름을 포함할 수도 있고, 레코드 번호나, 또는 무엇이든 추가할 수 있습니다.
*   `*value`: 레코드의 내용.
*   `*produced`: `true`로 설정.

파일의 끝(EOF)에 도달하면, `*at_end`를 `true`로 설정합니다. 어떤 경우에도 `Status::OK()`를 
반환합니다. 만약 에러가 발생하면, 매개변수를 변경하지 않고
[`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)
에 있는 도움말 함수중 하나를 단순히 반환하는데 사용합니다.

다음으로 여러분은 실제로 리더 오퍼레이션(op)을 만들 것입니다. 여러분이 
[오퍼레이션을 추가하는 방법](../../how_tos/adding_an_op/index.md)에 익숙하다면 도움이 될 겁니다.
주요 단계는 다음과 같습니다:

*   오퍼레이션(op) 등록.
*   `OpKernel`을 정의하고 등록.

오퍼레이션(op) 등록은, 
[`tensorflow/core/framework/op.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op.h)에
정의되어 있는 `REGISTER_OP`를 호출해서 사용할 수 있습니다.
리더 오퍼레이션(ops)은 다른 입력을 받을 수 없고 항상 `Ref(string)` 타입의 단일 출력만 합니다.
항상 `SetIsStateful()`를 호출해야 하고, `container`와 `shared_name` 속성(attrs)은 문자열을 갖습니다.
선택적으로 configuration에 대한 추가적인 속성을 정의하거나 `Doc`에 문서를 포함할 수 있습니다. 예를 들어,
[`tensorflow/core/ops/io_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/ops/io_ops.cc)
를 예로 들어봅시다.

```c++
#include "tensorflow/core/framework/op.h"

REGISTER_OP("TextLineReader")
    .Output("reader_handle: Ref(string)")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .Doc(R"doc(
A Reader that outputs the lines of a file delimited by '\n'.
)doc");
```

`OpKernel`를 정의하기 위해, 리더(Readers)는 
[`tensorflow/core/framework/reader_op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_op_kernel.h)
에 정의된 `ReaderOpKernel`에서 shortcut of descending을 사용할 수 있고, `SetReaderFactory`를 호출하는 생성자를 구현합니다.
클래스를 정의하고 난 후에, `REGISTER_KERNEL_BUILDER(...)`을 사용해서 등록해야 합니다.
속성이 없는 예제:

```c++
#include "tensorflow/core/framework/reader_op_kernel.h"

class TFRecordReaderOp : public ReaderOpKernel {
 public:
  explicit TFRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory([this, env]() { return new TFRecordReader(name(), env); });
  }
};

REGISTER_KERNEL_BUILDER(Name("TFRecordReader").Device(DEVICE_CPU),
                        TFRecordReaderOp);
```

속성이 있는 예제:

```c++
#include "tensorflow/core/framework/reader_op_kernel.h"

class TextLineReaderOp : public ReaderOpKernel {
 public:
  explicit TextLineReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int skip_header_lines = -1;
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_header_lines", &skip_header_lines));
    OP_REQUIRES(context, skip_header_lines >= 0,
                errors::InvalidArgument("skip_header_lines must be >= 0 not ",
                                        skip_header_lines));
    Env* env = context->env();
    SetReaderFactory([this, skip_header_lines, env]() {
      return new TextLineReader(name(), skip_header_lines, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TextLineReader").Device(DEVICE_CPU),
                        TextLineReaderOp);
```

마지막 단계는 파이썬 wrapper를 추가하는 것입니다. 
[`tensorflow/python/user_ops/user_ops.py`](https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py)
에 있는 `tensorflow.python.ops.io_ops`를 임포트할 수 있고, 
[`io_ops.ReaderBase`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py)의 서브 클래스를 추가할 수 있습니다.

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import io_ops

class SomeReader(io_ops.ReaderBase):

    def __init__(self, name=None):
        rr = gen_user_ops.some_reader(name=name)
        super(SomeReader, self).__init__(rr)


ops.NoGradient("SomeReader")
ops.RegisterShape("SomeReader")(common_shapes.scalar_shape)
```

몇 가지 예제는 
[`tensorflow/python/ops/io_ops.py`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py).
에서 확인할 수 있습니다.

## 레코드 형식 오퍼레이션 작성

일반적으로 이것은 스칼라 문자열 레코드를 입력으로 받고, 
[오퍼레이션(Op)을 추가하는 지침](../../how_tos/adding_an_op/index.md)
에 따르는 보통의 오퍼레이션(op)입니다. 선택적으로 스칼라 문자열 키를 입력하고, 적절하지 않은 형식의
데이터는 에러 메시지에 포함되어 리포팅 됩니다.
그런 방식으로 사용자들은 잘못된 데이터가 어디서 발생했는지 더 쉽게 추적할 수 있습니다.

디코딩 레코드에 대한 유용한 오퍼레이션(Ops)의 예:

*   [`tf.parse_single_example`](../../api_docs/python/io_ops.md#parse_single_example)
    (and
    [`tf.parse_example`](../../api_docs/python/io_ops.md#parse_example))
*   [`tf.decode_csv`](../../api_docs/python/io_ops.md#decode_csv)
*   [`tf.decode_raw`](../../api_docs/python/io_ops.md#decode_raw)

특정 레코드 형식을 디코딩하기 위해 여러 개의 오퍼레이션(Ops)을 사용하는 것은 유용합니다.
예를 들어, 여러분은 [`tf.train.Example`에 protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)를 통해
문자열로 저장 된 이미지를 가질 수 있습니다. 이미지의 형식에 따라, 
[`tf.parse_single_example`](../../api_docs/python/io_ops.md#parse_single_example) 오퍼레이션(op)과
[`tf.image.decode_jpeg`](../../api_docs/python/image.md#decode_jpeg),
[`tf.image.decode_png`](../../api_docs/python/image.md#decode_png), 또는
[`tf.decode_raw`](../../api_docs/python/io_ops.md#decode_raw)를 호출해서 대응하는 출력을 얻을 수 있을 겁니다.
`tf.decode_raw`의 출력을 가져가는 것이 일반적이고 
[`tf.slice`](../../api_docs/python/array_ops.md#slice)와 
[`tf.reshape`](../../api_docs/python/array_ops.md#reshape)의 조각을 추출하는 데 사용합니다.
