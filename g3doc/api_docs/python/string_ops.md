<!-- This file is machine generated: DO NOT EDIT! -->

# 문자열

참고 : `Tensor`를 인자로 받는 함수들은 [`tf.convert_to_tensor`]((framework.md#convert_to_tensor))의 인자로 들어갈 수 있는 값들 또한 받을 수 있습니다.

[TOC]

## 해싱 (Hashing)

문자열 해싱 연산은 문자열 입력 텐서를 받으며 각 원소를 하나의 정수와 매핑합니다.

- - -

### `tf.string_to_hash_bucket_fast(input, num_buckets, name=None)` {#string_to_hash_bucket_fast}

인풋 텐서의 각 문자열을 버킷의 갯수로 모듈러 함수를 적용한 해시로 변환합니다.

해시함수는 프로세스내에서 문자열의 내용에 의해 결정되며, 절대 바뀌지 않습니다. 그러나 이는 암호화에는 적절하지 않습니다. 이 함수는 CPU 시간이 부족하고 입력은 믿을 수 있거나 중요하지 않을 때 사용할 수 있습니다. 이 함수는 공격자가 모든 해시를 같은 버킷으로 보내버리도록 입력을 구성할 수 있다는 위험이 있습니다. 이러한 문제를 방지하기 위해선, `tf.string_to_hash_bucket_strong`라는 강력한 해시함수를 사용하십시오.

##### 인자:


*  <b>`input`</b>: `string`타입의 `Tensor`. 해시 버킷에 할당되는 문자열.
*  <b>`num_buckets`</b>: 1보다 같거나 큰 `int` 데이터. 버킷의 갯수.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `int64`타입의 텐서. 입력 `string_tensor`와 같은 shape을 가진 텐서.


- - -

### `tf.string_to_hash_bucket_strong(input, num_buckets, key, name=None)` {#string_to_hash_bucket_strong}

인풋 텐서의 각 문자열을 버킷의 갯수로 모듈러 함수를 적용한 해시로 변환합니다.

해시함수는 프로세스내에서 문자열의 내용에 의해 결정됩니다. 이 해시함수는 키를 가진 해시함수로 `key`라는 속성이 해시함수의 키를 정의합니다. `key`는 2개의 원소를 가진 배열입니다.

강력한 해시는 입력이 악의적일 수 있을 경우에 중요합니다. 가령, 추가적인 컴포넌를 가지는 URL이 있습니다. 공격자는 서비스 거부 공격이나 결과값을 왜곡하기 위해 입력값들이 같은 버킷으로 해시되도록 시도할 것입니다. 강력한 해시는 입력들을 같은 버킷으로 해싱하는 것이 불가능하지 않을 경우 이를 어렵게하여 방지할 수 있습니다. 이는 tf.string_to_hash_bucket_fast보다 약 4배 높은 계산 비용이 듭니다.

##### 인자:


*  <b>`input`</b>: `string`타입의 `Tensor`. 해시 버킷에 할당되는 문자열.
*  <b>`num_buckets`</b>: 1보다 같거나 큰 `int` 데이터. 버킷의 갯수.
*  <b>`key`</b>: `ints`의 리스트. 키를 가진 해시함수를 위한 키값으로 두 개의 uint64 타입의 원소를 가진 리스트로써 전달됩니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `int64`타입의 텐서. 입력 `string_tensor`와 같은 shape을 가진 텐서.


- - -

### `tf.string_to_hash_bucket(string_tensor, num_buckets, name=None)` {#string_to_hash_bucket}

인풋 텐서의 각 문자열을 버킷의 갯수로 모듈러 함수를 적용한 해시로 변환합니다.

해시함수는 프로세스내에서 문자열의 내용에 의해 결정됩니다.

해시함수는 수시로 변경될 수 있습니다.

##### 인자:


*  <b>`string_tensor`</b>: `string`타입의 `Tensor`.
*  <b>`num_buckets`</b>: 1보다 같거나 큰 `int` 데이터. 버킷의 갯수.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `int64`타입의 텐서. 입력 `string_tensor`와 같은 shape을 가진 텐서.



## 결합 (Joining)

문자열 결합 연산은 새로운 하나의 문자열 텐서를 생성하기위해 입력 문자열 텐서들의 원소를  하나로 잇습니다.

- - -

### `tf.reduce_join(inputs, reduction_indices, keep_dims=None, separator=None, name=None)` {#reduce_join}

주어진 차원에서 문자열 텐서를 결합합니다.

`[d_0, d_1, ..., d_n-1]`의 shape을 갖는 주어진 문자열 텐서의 차원에서 문자열을 결합합니다. 인자로 들어온 구분자 (기본값: 빈 문자열)를 가지고 입력 문자열들을 결합하여 생성된 새로운 텐서를 반환합니다. 음수 인덱스는 뒤에서부터 카운팅되며 `-1`은 `n - 1`과 동일합니다. 빈 `reduction_indices`를 전달하면 문자열들을 선형 인덱스 순서대로 결합하고 스칼라 문자열을 반환합니다.

예시:

```
# 텐서 `a`는 [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> ["acbd"]
tf.reduce_join(a, [1, 0]) ==> ["abcd"]
tf.reduce_join(a, []) ==> ["abcd"]
```

##### 인자:


*  <b>`inputs`</b>: `string`타입의 `Tensor`. 결합하고자 하는 입력값. 모든 감소된 인덱스들은 반드시 0이 아닌 크기를 가져야합니다.
*  <b>`reduction_indices`</b>: `int32`타입의 `Tensor`입니다. 차수를 줄이기위한 차원입니다. 지정된 순서에따라 차원이 감소됩니다. 만약 `reduction_indices`이 `1`보다 큰 랭크값을 가지면, 이는 1차원이 됩니다. `reduction_indices`를 생략하는건 `[n-1, n-2, ..., 0]`을 전달하는것과 같은 의미입니다. 음수 인덱스는 `-n`부터 `-1`까지 가능합니다.
*  <b>`keep_dims`</b>: 옵션이며 `bool`타입의 데이터입니다. 기본값은 `False`입니다. 만약 `True`일 경우, 길이가 `1`인 감소된 차원을 유지합니다.
*  <b>`separator`</b>: 옵션이며 `string`타입의 데이터입니다. 결합할 때 구분자로 사용됩니다.
*  <b>`name`</b>: 연산의 명칭 (선택사항).

##### 반환값:

  `string`타입의 `Tensor`. 감소된 차원만큼 제거되거나 `keep_dims`에 따라 1로 설정된 입력과 같은 shape을 가집니다.
