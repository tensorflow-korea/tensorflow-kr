# TensorFlow를 활용한 대형 스케일 선형 모델

tf.learn API는 TensorFlow에서 선형 모델을 사용하기 위한 (다른 것들중에서도) 풍부한 도구들을 제공합니다. 이 문서는 이러한 툴들에 대한 개요를 제공합니다. 이 문서에선 다음의 것들을 설명합니다.

   * 선형 모델이 무엇인지.
   * 왜 선형 모델을 사용하는지.
   * tf.learn이 TensorFlow에서 선형 모델을 얼마나 쉽게 구축할 수 있는지.
   * 양 쪽의 장점을 얻기 위해  tf.learn을 사용하여 어떻게 선형 모델과 딥러닝을 결합할 수 있는지.

tf.learn 선형 모델 도구가 당신에게 유용한지 아닌지를 결정하려면 이 문서를 읽어보십시오. 그리고 [Linear Models tutorial](wide/)를 시도해보십시오. 이 개요는 튜토리얼의 코드 샘플을 사용하지만, 튜토리얼은 코드를 더욱 상세하게 파고듭니다.

이 개요를 이해하기 위해선 머신 러닝의 기초 개념에 친숙해지는것과 [tf.learn](../tflearn/)를 보는것이 도음이 될 것입니다.

[TOC]

## 선형 모델이란?

*선형 모델*은 예측을 만들기위해 피쳐들의 단일 가중치 합을 사용합니다.
예를 들어, 인구의 나이, 교육 기간, 그리고 주별 근무시간에 대한 [data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)를 가지고 있다면 가중치 합이 개인의 급여를 추정하도록 숫자 각 각에 대한 가중치를 학습할 수 있습니다.

몇 가지 선형 모델은 가중치 합을 더욱 편리한 형태로 변환할 수 있습니다. 예를 들면, *로지스틱 회귀*는 출력값을 0과 1사이의 값으로 바꾸기위해 가중치 합을 로지스틱 함수로 연결합니다. 그러나 여전히 입력 피쳐들에 대한 단 하나의 가중치만 가지고 있습니다. 

## 왜 선형 모델을 사용하는가?

최근 연구에서 다중 레이어를 가진 매우 복잡한 신경망의 강력함이 입증되고 있는 시점에  왜 이러한 매우 단순한 모델을 사용할까요?

선형 모델은:

   * 깊은 신경망에 비해 학습이 빠릅니다.
   * 매우 큰 피쳐 집합에서도 잘 동작합니다.
   * 학습 속도와 귀찮은 작업등이 많이 필요하지 않는 알고리즘으로 훈련이 가능합니다.
   * 신경망에 비해 인터프리팅과 디버깅을 매우 쉽게 할 수 있습니다. 예측에 가장 큰 영향을 주는 피쳐가 무엇인지 찾기 위해 각 피쳐들에게 할당된 가중치들을 검사할 수 있습니다.
   * 머신러닝을 배우는데에 훌륭한 시작 지점을 제공합니다. (처음 머신러닝을 배우는데 적합합니다.)
   * 산업에서 널리 사용되고 있습니다.

## tf.learn이 어떻게 선형 모델을 구축하는데 도움을 주는가?

당신은 특별한 API의 도움 없이도 TensorFlow의 스크래치에서 선형 모델을 구축할 수 있습니다. 하지만 tf.learn은 효율적인 대형 스케일의 선형 모델을 쉽게 구축할 수 있는 몇 가지 도구를 제공합니다.

### 피쳐 컬럼과 변환(transformations)

선형 모델을 설계하는 대다수의 작업은 로우(raw) 데이터를 적당한 입력 피쳐들로 변환하는 작업들로 이루어집니다. tf.learn은 이러한 변환들을 가능하게하기 위해 `FeatureColumn` 추상화를 사용합니다.

`FeatureColumn`은 데이터에서 하나의 피쳐를 나타냅니다. `FeatureColumn`은 'height'와 같은 양적 수치를 나타낼 수도 있고, {'blue', 'brown', 'green'}와 같은 이산 확률 집합에서 뽑혀진 값인 'eye_color'와 같은 카테고리를 나타낼 수도 있습니다.

'height'와 같은 *연속적인 피쳐*와 'eye_color'와 같은 *카테고리성 피쳐*에서 데이터의 단일값은 모델로 입력되기 전에 숫자들의 시퀀스로 변환이 될 것입니다. `FeatureColumn` 추상화는 이러한 사실에도 불구하고 피쳐를 하나의 의미있는 단위로써 조작하도록 합니다. 당신은 변환을 지정할 수 있으며  당신이 모델에 넣을 텐서의 특정 인덱스를 처리하지않고 포함시킬 피쳐들을 선택할 수 있습니다.

### 희소 컬럼

선형 모델에서 카테고리성 피쳐들은 일반적으로 각 각의 가능한 값이 인덱스나 아이디를 가지고있는 희소 벡터로 변환됩니다. 예를 들면, 만약 `eye_color`를 길이가 3인 벡터로써 표현할 수 있는 딱 3가지의 가능한 eye_color가 있다고 해봅시다. 그러면 'brown'은 [1, 0, 0], 'blue'는 [0, 1, 0], 'green'은 [0, 0, 1]로 표현될 수 있습니다. 이 벡터들은 가능한 값들이 매우 많아질 경우(가령 모든 영어단어 라던지), 많은 제로값을 가지면서 매우 길어질 수 있기 때문에  "희소(sparse)"라고 부릅니다.

tf.learn 선형 모델을 사용하기 위해서 희소 컬럼을 사용할 필요는 없지만, 선형 모델의 강점중 하나는 매우 큰 희소 벡터를 처리하는 능력입니다. 희소 피쳐들은 tf.learn 선형 모델 도구의 가장 기본적인 사용 사례입니다.

#### 희소 컬럼 코드화

`FeatureColumn`는 카테고리성 값들의 벡터로의 변환을 자동으로 처리합니다. 다음의 코드를 보세요:

```python
eye_color = tf.contrib.layers.sparse_column_with_keys(
  column_name="eye_color", keys=["blue", "brown", "green"])
```

`eye_color`는 원데이터의 컬럼명입니다.

카테고리성 피쳐들의 모든 가능한 값들을 알 수 없는 경우에도 `FeaturColumn`을 생성할 수 있습니다. 이 경우엔 피쳐값들에게 인덱스를 할당하기위해 해쉬 함수를 사용하는  `sparse_column_with_hash_bucket()`을 사용할 수 있습니다.

```python
education = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "education", hash_bucket_size=1000)
```

#### 피쳐 교차

선형 모델은 피쳐들을 나누기위해 독립적인 가중치를 할당하기 때문에, 특정한 피쳐들의 조합의 상대적인 중요성은 학습할 수 없습니다. 만약 'favorite_sport'피쳐와 'home_city'피쳐를 가지고 있으며 어떤 사람이 빨간 옷을 좋아하는지 아닌지를 예측하려고 할 때, 당신의 선형 모델은 특히 빨간 옷을 좋아하는 St. Louis의 야구 팬들은 학습할 수 없을 것입니다.

당신은 'favorite_sport_x_home_city'라는 새로운 피쳐를 생성함으로써 이러한 한계를 극복할 수 있습니다. 주어진 사람에 대한 이 피쳐의 값은 단순히 두 개의 원래 피쳐값들을 하나로 잇는것입니다. 이러한 종류의 피쳐 조합을  *피쳐 교차(feature cross)*라고 부릅니다.

`crossed_column()`메서드는 피쳐 교차 생성을 쉽게 만들어줍니다.

```python
sport = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "sport", hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "city", hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column(
    [sport, city], hash_bucket_size=int(1e4))
```

#### 연속적인 컬럼

연속적인 피쳐는 다음과 같이 지정할 수 있습니다:

```python
age = tf.contrib.layers.real_valued_column("age")
```

비록 하나의 실수이기는 하지만, 연속적인 피쳐는 종종 모델로 바로 입력될 수 있으며, 뿐만 아니라 tf.learn은 이러한 종류의 컬럼을 위해 유용한 변환을 제공합니다.

##### 버킷화 (Bucketization)

*버킷화(Bucketization)*는 연속적인 컬럼을 카테고리성 컬럼으로 변환합니다. 이 변환은 연속적인 피쳐를 피쳐 교차에서 사용하도록 하거나 특별한 중요성을 가진 특정한 값 범위의 케이스를 학습하도록 합니다.

버킷화는 가능한 값들의 범위를 버킷이라 불리우는 부분 범위로 나눕니다. 

```python
age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

버킷은 그 값이 해당 값을 위한 카테고리 라벨이 됩니다.

#### 입력 함수

`FeatureColumn`은 모델의 입력 데이터를 위해, 어떻게 데이터를 표현하고 변환하는지를 나타내는 명세를 제공합니다. 그러나 데이터 그 자체를 제공하지는 않습니다. 데이터는 입력 함수를 통해 제공해줘야 합니다.

입력 함수는 반드시 텐서들의 딕셔너리를 반환해야합니다. 각 키는 `FeatureColumn`의 이름을 가리킵니다. 각 키의 값은 모든 데이터 인스턴스에 대한 피쳐의 값들을 포함하는 텐서입니다. 입력 함수의 예시는 [linear
models tutorial code](
https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py?l=160)에서 `input_fn`을 보십시오.

`fir()`과 `evaluate()`로 전달되는 입력 함수는 다음 섹션에서 보게될 훈련과 테스팅의 시작을 호출합니다.

### 선형 추정량

tf.learn의 추정량 클래스는 통합된 훈련과 회귀와 분류 모델을 위한 평가 하네스를 제공합니다. 이는 훈련과 평가 루프들의 상세한것들을 다루며 사용자는 모델 입력과 아키텍쳐에만 집중할 수 있도록 해줍니다.

선형 추정량을 구축하기 위해선 각 각 분류와 회귀를 위해 `tf.contrib.learn.LinearClassifier` 추정량 또는 `tf.contrib.learn.LinearRegressor` 추정량을 사용할 수 있습니다.

모든 tf.learn의 추정량에 대해, 실행은 단지 다음과 같이 하면됩니다.

   1. 추정량 클래스의 인스턴스를 만듭니다. 두 개의 선형 추정량 클래스를 위해선, `FeatureColumn`의 리스트를 생성자에 전달하면 됩니다.
   2. 훈련을 위해 추정량의 `fit()`메서드를 호출합니다.
   3. 어떻게 수행되는지 보기 위해 추정량의 `evaluate()`메서드를 호출합니다.

예시:

```python
e = tf.contrib.learn.LinearClassifier(feature_columns=[
  native_country, education, occupation, workclass, marital_status,
  race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  model_dir=YOUR_MODEL_DIRECTORY)
e.fit(input_fn=input_fn_train, steps=200)
# 한 단계 평가하기 (테스트 데이터를 통해 전달)
results = e.evaluate(input_fn=input_fn_test, steps=1)

# 평가에 대한 통계를 출력합니다.
for key in sorted(results):
    print "%s: %s" % (key, results[key])
```

### 넓고 깊은 학습

tf.learn API는 또한 선형 모델과 깊은 신경망를 함께 훈련시킬 수 있도록하는 추정량 클래스를 제공합니다. 이 새로운 접근법은 신경망의 일반화 능력과 선형 모델의 키 피쳐들을 기억할 수 있는 능력을 결합합니다. 이러한 "넓고 깊은" 종류의 모델을 생성하려면 `tf.contrib.learn.DNNLinearCombinedClassifier` 사용하세요.

```python
e = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```

더 상세한 내용은 [Wide and Deep Learning tutorial](../wide_and_deep/)를 보십시오.
