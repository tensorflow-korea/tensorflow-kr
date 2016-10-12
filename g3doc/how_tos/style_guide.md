# 텐서플로우 스타일 가이드

텐서플로우의 개발자들과 사용자들이 따라야 할 코딩 스타일에 대한 내용입니다. 코드의 가독성을 좋게 하고, 에러를 줄이며, 일관성을 높이는 것이 목적입니다.

[TOC]

## 파이썬 스타일

들여쓰기 2칸이라는 것을 제외하면, 일반적으로
[PEP8 Python style guide](https://www.python.org/dev/peps/pep-0008/) 가이드를 따릅니다.

## 파이썬 2, 3 호환

* 모든 코드는 파이썬 2 및 3과 호환되어야 합니다.

* 모든 파이썬 파일들에 다음 라인들이 들어 있어야 합니다:

```
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
```

* 코드의 호환성을 위해 `six` 를 사용하십시요. (예: `six.range`).


## 베이젤(Bazel) BUILD 규칙

텐서플로우는 Bazel 빌드 시스템을 사용하며, 그에 따라 다음의 조건들이 요구됩니다:

* 모든 BUILD 파일은 다음의 헤더를 포함해야 합니다:

```
# Description:
# <...>

package(
    default_visibility = ["//visibility:private"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])
```

* 모든 BUILD 파일의 마지막에는 다음 내용이 있어야 합니다:

```
filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//third_party/tensorflow:__subpackages__"],
)
```

* 새 BUILD 파일을 추가하는 경우, 다음의 내용을 `tensorflow/BUILD` 파일에 추가하여 `all_opensource_files` 타겟이 되도록 합니다.

```
"//third_party/tensorflow/<directory>:all_files",
```

* 모든 파이썬 BUILD 타겟(라이브러리와 테스트)에 다음의 내용을 추가하십시요:

```
srcs_version = "PY2AND3",
```


## 텐서

* 배치(batch)를 다루는 오퍼레이션은 텐서의 첫번째 차원을 배치의 차원으로 간주할 수 있습니다.

## 파이썬 오퍼레이션

*파이썬 오퍼레이션* 이란, 입력 텐서와 파라미터를 받아, 그래프의 일부를 생성하고 출력 텐서를 리턴하는 함수를 의미합니다.

* 텐서값들은 처음에 오는 인자들로 받아야 하며, 일반적인 파이썬 파라미터들은 그 다음 인자들로 받습니다. 마지막 인자는 `name`인데, 디폴트 값은 `None`입니다. 오퍼레이션이 그래프 콜렉션들에 어떠한 `Tensor`들을 저장할 필요가 있는 경우에는, 콜렉션들의 이름을 인자에 담아서, `name` 인자의 바로 앞순서로 받아야 합니다.

* 텐서값을 담은 인자들은 하나의 텐서이거나 반복가능한 텐서이어야 합니다. 예: "텐서 또는 텐서의 리스트" 는 지나치게 광범합니다. `assert_proper_iterable`를 참조하십시요.

* 텐서를 인자로 받아들이는 오퍼레이션이 C++ 오퍼레이션들을 사용하는 경우, `convert_to_tensor`를 호출하여 非텐서 입력들을 텐서들로 변환해야 합니다. 문서에서 해당 인자들은 여전히 `Tensor` 라는 특별한 자료형 객체로 표현된다는 점을 기억하십시요.

* 각각의 파이썬 오퍼레이션들은 아래와 같은 `op_scope`를 가져야 합니다. 입력 텐서들의 리스트와 `name`, 그리고 그 op의 디폴트 네임을 인자로 전달하십시요.

* 오퍼레이션들에 선언되어 있는 인자들과 리턴값들은, 상세한 파이썬 주석을 통하여 그 의미와 자료형이 설명되어야 합니다. 가능한 형태, 자료형, 랭크(텐서의 계수)도 주석에 명시되어야 합니다.
 [도큐멘테이션 상세 설명 보기](documentation/index.md)

* 사용 편의를 높이기 위하여, 해당 오퍼레이션의 입력값과 출력값이 포함되어 있는 사용 예를 Example 항목에 포함해야 합니다.

예:

    def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
              output_collections=(), name=None):
    """My operation that adds two tensors with given coefficients.

    Args:
      tensor_in: `Tensor`, input tensor.
      other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
      my_param: `float`, coefficient for `tensor_in`.
      other_param: `float`, coefficient for `other_tensor_in`.
      output_collections: `tuple` of `string`s, name of the collection to
                          collect result of this op.
      name: `string`, name of the operation.

    Returns:
      `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

    Example:
      >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
                output_collections=['MY_OPS'], name='add_t1t2')
      [2.3, 3.4]
    """
    with tf.op_scope([tensor_in, other_tensor_in], name, "my_op"):
      tensor_in = tf.convert_to_tensor(tensor_in)
      other_tensor_in = tf.convert_to_tensor(other_tensor_in)
      result = my_param * tensor_in + other_param * other_tensor_in
      tf.add_to_collections(output_collections, result)
      return result

실제 사용시:

    output = my_op(t1, t2, my_param=0.5, other_param=0.6,
                   output_collections=['MY_OPS'], name='add_t1t2')


## 레이어

*레이어*는 변수 생성 기능과 하나 이상의 다른 그래프 오퍼레이션들을 합친 파이썬 오퍼레이션입니다. 정규 파이썬 오퍼레이션의 요구 조건들을 따르십시요.

* 레이어가 하나 이상의 변수를 생성하는 경우, 레이어 함수는 다음의 인자들을 나열된 순서대로 받아들여야 합니다:
  - `initializers`: 변수들의 초기화 방법을 명시하는 경우에 선택적으로 사용.
  - `regularizers`: 변수들의 정측화(regularization) 방법을 명시하는 경우에 선택적으로 사용.
  - `trainable`: 변수들이 학습 가능한지의 여부를 컨트롤 함.
  - `scope`: `VariableScope` 객체로서, 변수들의 범위를 나타냄.
  - `reuse`: 변수가 범위(scope)내에 있는 경우, 재사용 할 것인지를 나타내는 `bool` 값.

* 전체 모델이 학습하는 중에, 학습하지 않는 레이어는 다음의 인자를 가져야 합니다:
  - `is_training`: 학습 그래프가 만들어져 있는지를 나타내기 위한 `bool` 인자.


예:

    def conv2d(inputs,
               num_filters_out,
               kernel_size,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalization_fn=add_bias,
               normalization_params=None,
               initializers=None,
               regularizers=None,
               trainable=True,
               scope=None,
               reuse=None):
      ... see implementation at tensorflow/contrib/layers/python/layers/layers.py ...

