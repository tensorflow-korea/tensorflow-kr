<!-- This file is machine generated: DO NOT EDIT! -->

# 테스팅

## 유닛 테스트

TensorFlow는 `unittest.TestCase`를 상속하고 TensorFlow 테스트와 관련된 메서드를 추가한 편리한 클래스를 제공합니다. 아래에 예시가 하나 있습니다.

```python
    import tensorflow as tf


    class SquareTest(tf.test.TestCase):

      def testSquare(self):
        with self.test_session():
          x = tf.square([2, 3])
          self.assertAllEqual(x.eval(), [4, 9])


    if __name__ == '__main__':
      tf.test.main()
```

`tf.test.TestCase`는 `unittest.TestCase`를 상속하지만 추가적인 메서드가 더 있습니다. 우리는 곧 이 메서드들에 대해 문서화를 할 것입니다.

- - -

### `tf.test.main()` {#main}

모든 유닛 테스트를 실행합니다.


## 유틸리티

- - -

### `tf.test.assert_equal_graph_def(actual, expected)` {#assert_equal_graph_def}

두 개의 `GraphDef`가 (대부분) 같은지 확인합니다.

두 `GraphDef`의 원형이 같은지를 비교하는데 버전과 노드의 순서, 속성 그리고 제어 입력은 무시합니다. 노드명은 두 그래프 사이의 노드를 매칭하는데 사용되기 때문에 노드의 네이밍은 일관되어야합니다.

##### 인자:


*  <b>`actual`</b>: 테스트를 위한 `GraphDef`.
*  <b>`expected`</b>: 예상값 `GraphDef`.

##### 예외:


*  <b>`AssertionError`</b>: 두 `GraphDef`가 매칭이 안될 경우 발생합니다.
*  <b>`TypeError`</b>: 둘 중 하나라도 `GraphDef`가 아닐 경우 발생합니다.


- - -

### `tf.test.get_temp_dir()` {#get_temp_dir}

테스트중 사용할 임시 디렉토리를 반환합니다.

테스트 후 디렉토리를 삭제할 필요가 없습니다.

##### 반환값:

  임시 디렉토리.

- - -

### `tf.test.is_built_with_cuda()` {#is_built_with_cuda}

TensorFlow의 CUDA (GPU) 지원 여부를 반환합니다.

## 그라디언트(Gradient) 확인

[`compute_gradient`](#compute_gradient)와 [`compute_gradient_error`](#compute_gradient_error)는 등록된 해석적 그라디언트와 그래프의 수치 미분의 비교를 수행합니다.

- - -

### `tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None)` {#compute_gradient}

이론 및 수치적 코비안을 계산한 후 반환합니다.

만약 `x`또는 `y`가 복소수이면, 코비안은 여전히 실수지만 대응되는 코비안 차원은 두 배가 될 것입니다. TensorFlow 그래프가  정칙(holomorphic)일 필요는 없기 때문에 비록 입력과 출력이 모두 복소수라고 해도 이는 필수적입니다. 그리고 이는 복소수로써 표현될 수 없는 그라디언트를 가질 것입니다. 예를 들면, `x`가 `[m]` shape을 갖는 복소수이고 `y`가 `[n]` shape을 갖는 복소수일 때, 각 코비안 `J`는 다음과 같은 `[m * 2, n * 2]` shape을 갖게될 것입니다.

    J[:m, :n] = d(Re y)/d(Re x)
    J[:m, n:] = d(Im y)/d(Re x)
    J[m:, :n] = d(Re y)/d(Im x)
    J[m:, n:] = d(Im y)/d(Im x)

##### 인자:


*  <b>`x`</b>: 텐서 또는 텐서들의 리스트.
*  <b>`x_shape`</b>: 정수형 튜플 또는 배열 형태의 x의 차원입니다. x가 리스트라면 이는 shape들의 리스트입니다.

*  <b>`y`</b>: 텐서.
*  <b>`y_shape`</b>: 정수형 튜플 또는 배열 형태의 y의 차원입니다.
*  <b>`x_init_value`</b>: (선택적인) "x"와 같은 shape을 가진 numpy 배열로 x의 초기값을 나타냅니다. 만약 x가 리스트라면, 이는 numpy 배열의 리스트여야합니다. 만약 값이 없다면, 함수는 초기값 텐서를 랜덤으로 선택할 것입니다.
*  <b>`delta`</b>: (선택적인) 섭동의 크기.
*  <b>`init_targets`</b>: 모델 파라미터 초기화를 실행하기 위한 타겟들의 리스트.
    TODO(mrry): 이 인자를 없앱니다.

##### 반환값:

  dy/dx에 대한 이론 및 수치적 코비안을 나타내는 두 개의 이차원 numpy 배열. 각 각은 x의 원소의 갯수인 "x_size"개의 행과 y의 원소의 갯수인 "y_size"개의 열을 가집니다. 만약 x가 리스트라면, 두 개의 numpy 배열의 리스트를 반환합니다.

- - -

### `tf.test.compute_gradient_error(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None)` {#compute_gradient_error}

그라디언트 오차를 계산합니다.

계산된 코비안과 수치적으로 추정된 코비안간의 dy/dx에 대한 최대 오차를 계산합니다.

이 함수는 연산들을 추가함으로써 전달된 텐서를 변경시키고 따라서 입력 텐서의 연산을 사용하는 컨슈머들을 바꿉니다.

이 함수는 현재 세션에 연산들을 추가합니다. GPU 같은 특정한 디바이스를 사용하여 오차를 계산하기 위해선 디바이스 설정을 위한 표준 메서드를 사용합니다. (예를 들면 with sess.graph.device()를 사용하거나 세션 생성자에 디바이스 함수를 설정)

##### 인자:


*  <b>`x`</b>: 텐서 또는 텐서들의 리스트.
*  <b>`x_shape`</b>: 정수형 튜플 또는 배열 형태의 x의 차원입니다. x가 리스트라면 이는 shape들의 리스트입니다.

*  <b>`y`</b>: 텐서.
*  <b>`y_shape`</b>: 정수형 튜플 또는 배열 형태의 y의 차원입니다.
*  <b>`x_init_value`</b>: (선택적인) "x"와 같은 shape을 가진 numpy 배열로 x의 초기값을 나타냅니다. 만약 x가 리스트라면, 이는 numpy 배열의 리스트여야합니다. 만약 값이 없다면, 함수는 초기값 텐서를 랜덤으로 선택할 것입니다.
*  <b>`delta`</b>: (선택적인) 섭동의 크기.
*  <b>`init_targets`</b>: 모델 파라미터 초기화를 실행하기 위한 타겟들의 리스트.
    TODO(mrry): 이 인자를 없앱니다.

##### 반환값:

  두 코비안간의 최대 오차값.


