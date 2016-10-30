# 텐서플로우 선형 모델 튜토리얼

이번 강의에서 우리는 이진 분류 문제를 사람에 나이, 성별, 교육, 그리고 직업(특성들)에 관한 인구조사 데이터를 가지고 한 사람의 연봉이 50,000불이 넘는지를 TensorFlow에 TF.Learn API를 사용해서 풀어 볼 것이다(목표 레이블). 우리는 **로지스틱 회귀** 모델을 주어진 개인들에 정보를 가지고 교육 할 것이고 모델은 개인의 연봉이 50,000달러 이상일 가능성으로 해석 될 수 있는 0과1 사이의 숫자를 출력한다.

## 설치

이번 튜토리얼 코드를 실행해보기 위해서:

1.  텐서플로우를 설치하지 않았다면 [텐서플로 설치](../../get_started/os_setup.md)

2.  [튜토리얼 코드](
    https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py) 다운로드.

3.  pandas 데이터 분석 라이브러리 설치.

 tf.learn에 pandas가 필수적인 요소는 아니지만 tf.learn이 판다를 지원하고 또한 이번 튜토리얼에서 판다를 하기 때문. pandas를 설치하기 위해서:

    1.  `pip` 설치:

       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```

    2. `pip`로 pandas 설치하기:

       ```shell
       $ sudo pip install pandas
       ```

    만약에 판다 설치에 어려움을 느낀다면, 판다 사이트에서 [설명]
(http://pandas.pydata.org/pandas-docs/stable/install.html) 을 참고하시오.

4. 이 튜토리얼에 설명된 선형모델을 훈련하기 위해 튜토리얼 코드를 아래의 명령어로 실행하시오:

   ```shell
   $ python wide_n_deep_tutorial.py --model_type=wide
   ```
코드가 어떻게 선형모델을 구축하는지 계속 읽어 보자.

## 인구조사 데이터 읽어보기

우리가 사용할 데이터 세트는 [소득 인구조사 데이터세트]
(https://archive.ics.uci.edu/ml/datasets/Census+Income). [훈련 데이터]
(https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) 그리고
[테스트 데이터](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test) 를 수동 또는 코드를 이용해서 내려받을 수 있습니다.

```python
# 텐서플로 선형 모델 튜토리얼

이번 강의에서 우리는 이진 분류 문제를 사람에 나이, 성별, 교육, 그리고 직업(특성들)에 관한 인구조사 데이터를 가지고 한 사람의 연봉이 50,000불이 넘는지를 TensorFlow에 TF.Learn API를 사용해서 풀어 볼 것이다(목표 레이블). 우리는 **로지스틱 회귀** 모델을 주어진 개인들에 정보를 가지고 교육할 것이고 모델은 개인의 연봉이 50,000달러 이상일 가능성으로 해석 될 수 있는 0과1 사이의 숫자를 출력한다.

## 설치

이번 튜토리얼 코드를 실행해보기 위해서:

1.  텐서플로우를 설치하지 않았다면 [텐서플로 설치](../../get_started/os_setup.md)

2.  [튜토리얼 코드](
    https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py) 다운로드.

3.  pandas 데이터 분석 라이브러리 설치.

 tf.learn에 pandas가 필수적인 요소는 아니지만 tf.learn이 판다를 지원하고 또한 이번 튜토리얼에서 판다를 하기 때문. pandas를 설치하기 위해서:

    1.  `pip` 설치:

       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```

    2. `pip`로 pandas 설치하기:

       ```shell
       $ sudo pip install pandas
       ```

    만약에 판다 설치에 어려움을 느낀다면, 판다 사이트에서 [설명]
(http://pandas.pydata.org/pandas-docs/stable/install.html) 을 참고하시오.

4. 이 튜토리얼에 설명된 선형모델을 훈련하기 위해 튜토리얼 코드를 아래의 명령어로 실행하시오:

   ```shell
   $ python wide_n_deep_tutorial.py --model_type=wide
   ```
코드가 어떻게 선형모델을 구축하는지 계속 읽어 보자.

## 인구조사 데이터 읽어보기

우리가 사용할 데이터 세트는 [소득 인구조사 데이터세트]
(https://archive.ics.uci.edu/ml/datasets/Census+Income). [훈련 데이터]
(https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) 그리고
[테스트 데이터](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test) 를 수동 또는 코드를 이용해서 내려받을 수 있습니다.

```python
import tempfile
import urllib
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)
```
CSV 파일들에 다운로드가 완료됐다면, [Pandas] (http://pandas.pydata.org/) 데이터프레임에 입력시켜 보자.

```python
import pandas as pd
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
```

이번 과제가 이진 분류 문제이기 때문에 수입이 50,000달러가 넘는다면 1을 그렇지 않다면 0에 값을 가지는 열의 이름이 "label"인 표를 만들 것이다.

```python
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
```

다음으로, 데이터프레임에서 어떤 열들이 목표 label을 예측하는 데 사용 될 수 있는지 살펴보자. 열들은 categorical 또는 continuous 두 타입으로 구분 될 수 있다.



*   만약에 값이 오직 유한집합 범주 안에 있을 때 **categorical** 열이라 불린다.
    예를 들어 사람에 국적(미국, 인도, 일본 등)이나 교육 수준(고등학교, 대학 등)이 categorical 열들이다.



*  만약에 값이 어떤 수치로 나올 수 있다면 **continuous** 열 이라 불린다. 예를 들어, 한 사람에 소득이 continuous열이다.


```python
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
```

수입 인구조사 데이터 세트에 나오는 열 리스트:

|열 이름    | 타입        | 설명                       | {.sortable}   |
| -------------- | ----------- | --------------------------------- |
| age            | Continuous  | 나이                               |
| workclass      | Categorical | 고용주 타입                          |
:                :             : (정부, 군대, 기업 등)                 :
| fnlwgt         | Continuous  | The number of people the census   |
:                :             : takers believe that observation   :
:                :             : represents (sample weight). This  :
:                :             : variable will not be used.        :
| education      | Categorical | 최고 학력                           |
| education_num  | Continuous  | 숫자 형식의 최고 학력                  |
| marital_status | Categorical | 혼인 여부                           |
| occupation     | Categorical | 직업                               |
| relationship   | Categorical | Wife, Own-child, Husband,         |
:                :             : Not-in-family, Other-relative,    :
:                :             : Unmarried.                        :
| race           | Categorical | 인종 ( 백인, Asian-Pac-Islander,    |
:                :             : Amer-Indian-Eskimo, Other, 흑인.   :
| gender         | Categorical | 성별 (남, 여)                       |
| capital_gain   | Continuous  | 기록된 양도 소득.                     |
| capital_loss   | Continuous  | 기록된 자본 손실.                     |
| hours_per_week | Continuous  | 주당 근무시간.                       |
| native_country | Categorical | 출생지                              |
| income         | Categorical | ">50K" 또는 "<=50K", 개인의 일년      |
:                :             : 수입이 5만불 이상인지 아닌지 뜻함         :


##데이터를 텐서들로 바꾸기

TF.Learn 모델을 구축 할 때, 입력 데이터는 Input Builder 함수에 의해서 명시된다.
이 builder 함수는 TF.Learn에 `fit` 이나 `evaluate` 와 같은 메소드들에게 넘겨 질때 까지 호출되지 않는다.
이 함수의 목적은 입력 데이터를 [Tensors](https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Tensor) 나
[SparseTensors](https://www.tensorflow.org/versions/r0.9/api_docs/python/sparse_ops.html#SparseTensor) 형태로 구성하기 위함에 있다.
더 구체적으로, Input Builder 함수는 다음과 같은 한 쌍을 반환한다:

1.  `feature_cols`:  A dict from feature column names to `Tensors` or
    `SparseTensors`.
2.  `label`: 라벨 열을 포함 하고 있는 `Tensor`.

`feature_cols`에 키들은 다음 부분에서 열을 구성하는 데 사용 될 것이다.

우리는 `fit` 과 `evaluate` 메소드들을 서로 다른 데이터로 호출 하고 싶으므로,
서로 같지만 단지 다른 데이터를 `input_fn`에 전달하는 input builder 함수인 `train_input_fn` 그리고 `test_input_fn`를 정의 했습니다.

여기서 눈여겨볼 것은 `input_fn`가 그래프 실행 중이 아니라 텐서플로 그래프를 생성하는 도중에 호출된다는 것입니다.
Input Builder가 반환하는 것은 입력 데이터를 대표하는 텐서플로 연산 기본단위인 `Tensor`(또는 `SparseTensor`)입니다.

우리 모델은 입력 데이터의 정숫값을 대표하는 *constant* 텐서로 나타낸다, 이 경우에는 `df_train` 나 `df_test`에 열에 값을 대표한다.
이 방법이 텐서플로에 데이터를 전달하는 가장 간단한 방법이다.
다른 심화한 방법으로는 파일이나 다른 데이터 소스를 대표하는 Input Reader를 만들어서 파일을 텐서플로가 그래프를 실행하는 동안에 읽어 나가는 것이다.

훈련 또는 테스트 데이터 프레임에 있는 각각의 continuous 열들은 밀집 데이터를 대표하기 좋은 `Tensor`로 변환될 것입니다.

우리는 categorical 데이터를 반드시 `SparseTensor`로 나타내야 합니다.
Categorical 데이터 포맷은 희소 데이터를 대표하기 좋습니다.

```python
import tensorflow as tf

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)
```

## 모델을 위한 선택과 공학 특징

올바른 특성 열 세트를 선택하고 만드는 것이 효과적인 모델 학습에 핵심입니다.

**feature column**은 기존 데이터 프레임(**기본 특성 열**)에 가공되지 않은 열중 이거나, 
하나 또는 여러게의 기본 열(**파생 특성 열**)에 변화를 기본으로 새롭게 생성된 열이다.

근본적으로, "feature column"은 목표 레이블을 예상하는 데 사용 가능한 추상적인 개념의 비가공 또는 파생 변수이다.

### 기본 Categorical 특성 열

categorical 특성을 위한 특성열을 정의 하기 위해서 우리는 TF.Learn API로 `SparseColumn` 생성할 수 있습니다.  
만약에 모든 열에 특성값 세트를 알고 있고 또한 몇 개 안된다면, `sparse_column_with_keys`를 사용할 수 있습니다.
리스트에 안에 각각의 키들은 0부터 시작해서 자동으로 증가하는 아이디가 할당된다.
예를 들어 우리는 아래와 같이 성별 열에 특성 문자열 "female"에게 숫자 아이디 1을 그리고 "male"에게는 1을 할당 할 수 있다

```python
gender = tf.contrib.layers.sparse_column_with_keys(
  column_name="gender", keys=["female", "male"])
```

만약에 가능한 값의 세트를 미리 알 수 없다면 어떻게 해야 하나 ? 문제없다.
우리는 `sparse_column_with_hash_bucket`을 대신 사용할 수 있다.

```python
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
```

`education`에 각 특성열에 가능한 값들은 훈련중에 정수 아이디로 헤시 되어질 것이다. 아래의 실례를 보자:

ID  | Feature
--- | -------------
... |
9   | `"학사"`
... |
103 | `"박사"`
... |
375 | `"석사"`
... |

어떤 방법으로 `SparseColumn`를 정의 하던지 특성 문자열들은 정해진 멥핑 또는 헤쉬를
정수 ID를 찾을 것이다. 여기서 헤쉬 충돌이 일어날 수 있지만, 모델의 질에 큰 영향을 끼치지는 않을 것이다. 내부적으로 `LinearModel` 클래스가 모델의 각 특성 ID의 모델 매개변수(모델 무게로 알려진)를 저장하는데 사용되는 `tf.Variable`의 생성과 측정을 책임지고 있다. 모델의 매개변수들은 우리가 나중에 배우게 될 모델 훈련 과정에서 알게 될 것입니다.

우리는 다른 categorical 특성들을 정의하기 위해 비슷한 기술을 사용할 것이다.

```python
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
```

### 기초 Continuous 특성 열

비슷하게, 모델에서 사용하고 싶은 각 continuous 특성열들에게 `RealValuedColumn`을 정의 할 수 있습니다.

```python
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
```

### 버킷화를 통해 Continuous 특성들을 범주화 하기

때때로 continuous 특성과 라벨의 관계는 비선형이다. 가상의 한 예와 같이, 한 사람의 수입이 사회생활 초기에는 나이와 함께 증가할 것이고, 어느 때가 되면 수입의 증가는 더뎌지고, 그리고 마침내 은퇴 후에는 수입이 줄어 들것이다. 모델이 아래의 3가지 경우 중 하나만 습득할 수 있으므로 이 시나리오에서 비가공 `age`를 실수특성 열로 사용하는 것은 좋은 선택이 아닐 것이다.

1.  수입이 나이의 증가에 따라 항상 같은 비율로 증가 (양의 상관관계)
2.  수입이 나이의 증가에 따라 항상 같은 비율로 감소 (음의 상관관계), 또는
3.  수입이 나이에 상관 없이 항상 같음(관계없음)

만약 우리가 수입과 각 나이 그룹과의 세밀한 상관관계를 학습하고 싶다면  **버킷화** 를 활용할 수 있다. 버킷화는 continuous 특성 전체를 연속적인 빈/버킷들의 세트로 나누고, 이후에 버킷의 값에 따라 원래의 수적 특성을 버킷 아이디(categorical feature)로 변환하는 과정이다. 그래서 우리는 `age`에 대해 `bucketized_column`을 정의 할 수 있다:

```python
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```
`boundaries`는 버킷 경계의 목록이다.

이 경우에 10개의 경계가 있고, 결과적으로 11개의 버킷 그룹이 생성된다 ( 0세부터 17까지, 18-24, 25-29…. 65세 이상).

### 다수의 열을 CrossedColumn으로 교차하기

각 기본 feature 열을 나눠 사용하는 것만으로는 데이터를 설명하는데 충분하지 않을 것이다.
예를 들어 교육과 레이블(수입 > 50,000)에 상관관계는 아마 직업에 따라 다를 것이다.
그렇기에 우리가 오직 한가지 모델 `education="학사"` and `education="석사"`의 무게를 학습한다면, 우리는 모든 경우의 교육-직업 조합 `education="학사" AND occupation="경영자"`
and `education="학사" AND occupation="수리공"`)의 차이를 알아 낼 수 없다.
다른 특성 조합들의 차이를 알기 위해서 우리는 **crossed feature columns** 을 모델에 부여할 수 있습니다.

```python
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
```

우리는 또한 `CrossedColumn`을 두 가지 이상의 열에 생성할 수 있습니다.
각 constituent 열은 기본 특성 열 categorical(`SparseColumn`)이나 버킷 화 된 실수 특성 열(`BucketizedColumn`), 심지어 다른 `CrossColumn`이 될 수 있습니다. 하나의 예:

```python
age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column(
  [age_buckets, race, occupation], hash_bucket_size=int(1e6))
```

## 로지스틱 회귀 모델 정의하기

입력 데이터를 가공 하고 모든 특성열들을 정의 한 다음, 이제 모든 것을 한자리에 모아 로지스틱 회귀 모델을 구축할 준비가 되었다.
이전 부분에서 우리는 아래의 특성열들을 포함한 여러 가지의 기본 그리고 파생 특성 열을 보았다.

*   `SparseColumn`
*   `RealValuedColumn`
*   `BucketizedColumn`
*   `CrossedColumn`

위의 모든 특성열들은 추상 클래스 `FeatureColumn`에 하위 클래스 들이며, 모델의 `feature_columns` 필드에 추가될 수 있습니다.

```python
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  model_dir=model_dir)
```

모델은 자동으로 feature들을 보지 않고도 예측을 제어 할 수 있는 편향 용어를 자동으로 학습할 것입니다(더 많은 설명을 위해서는 "로지스틱 회귀가 어떻게 작동하나" 보시오). 학습된 모델 파일은 `model_dir`에 저장될 것입니다.

## 모델을 훈련, 평가하기

모델에 모든 특성을 추가한 다음 어떻게 실제로 모델을 훈련 시키는지 알아보자.
모델을 훈련하는 것은 TF.Learn API를 사용하면 한 줄이면 된다.

```python
m.fit(input_fn=train_input_fn, steps=200)
```
모델을 훈련 시킨 뒤 모델이 얼마나 홀드 아웃 데이터의 라벨을 잘 예측하는지 평가해 볼 수 있다

```python
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])
```

첫 번째 줄의 결과는 정확도 83.6%를 뜻하는 `accuracy: 0.83557522`와 같이 나올 것이다.
당신이 더 좋은 결과를 낼 수 있는지 자유롭게 더 많은 기능을 시도해보고 변경해봐라

만약에 처음부터 끝까지 작동하는 예제를 보고 싶다면 우리의 [예제코드를](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py) 내려받고, `model_type` 플래그를 `wide`로 설정하세요.

## 과적화를 피하기위해 정규화 추가하기

정규화는 **과적화**를 피하려고 사용되는 기술이다. 과적화는 모델이 훈련에 사용했던 데이터로는 잘 작동하지만, 처음 보는 데이터에 대해서는 안 좋은 결과를 보일 때 일어난다. 과적화는 모델이 너무 과도하게 복잡할 때 일어난다. 예를 들어 관찰한 자료에 비해 매개변수가 너무 많을 경우이다. 정규화는 모델의 복잡성을 제어 할 수 있게 해주고, 처음 접하는 데이터에 대해 좀 더 일반화가 가능하게 해준다.

선형모델 라이브러리에 당신은 L1과 L2 정규화 들을 다음과 같이 모델에 추가할 수 있습니다:

```
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir)
```

L1과 L2 정규화의 한 가지 중요한 차이점은 L1 정규화는 모델의 무게를 0에 있게 하려고 하고 sparser 모델을 만들지만, L2 정규화는 모델에 무게를 0에 가깝게 만들려고 노력한다 하지만 꼭 0일 필요는 없습니다. 그렇기에 당신이 만약 L1 정규화를 강화한다면 많은 모델에 무게가 0일 것이라 작은 모델을 가지게 될 것입니다. 정규화는 feature 크기가 크지만 흩어져 있는 경우와 자원이 제한적이라 아주 큰 모델을 실행하기 힘든 경우에 종종 바람직한 방법 입니다.

실제로는 과적화와 모델의 크기를 조절 하기에 최고의 매개변수를 찾기 위해 L1과 L2의 강도를 다양하게 시도해봐야 합니다.

## 로지스틱 회귀가 어떻게 작동하는가

만약 로지스틱 회귀 모델에 익숙하지 않다면 여기서 잠시 멈춰 로지스틱 회귀모델이 실제로 어떻게 생겼는지 보도록 하자. 우리는 라벨을 $$Y$$로, 관찰된 feature들의 세트를 feature 벡터 $$\mathbf{x}=[x_1, x_2, ..., x_d]$$로 표현할 것이다. 우리는 수입이 50,000달러가 넘는다면 $$Y=1$$ 그렇지 않다면 $$Y=1$$ 정의 할 것 입니다. 로지스틱 회귀에서 주어진 features $$\mathbf{x}$$에 대해 라벨이 양수($$Y=1$$)가 될 가능성을 나타내는 것은:

$$ P(Y=1|\mathbf{x}) = \frac{1}{1+\exp(-(\mathbf{w}^T\mathbf{x}+b))}$$

where $$\mathbf{w}=[w_1, w_2, ..., w_d]$$ are the model weights for the features $$\mathbf{x}=[x_1, x_2, ..., x_d]$$.

$$\mathbf{w}=[w_1, w_2, ..., w_d]$$는 features $$\mathbf{x}=[x_1, x_2, ..., x_d]$$의 모델 무게이다. $$b$$는 정수이며 종종 모델의 **편향** 이라고 불린다. 공식은 두 parts-A 선형모델과 로지스틱 함수로 구성 되어 있다.

*   **선형 모델**: 먼저, 우리는 결과가 입력 features $$\mathbf{x}$$ 함수인 선형 모델 $$\mathbf{w}^T\mathbf{x}+b = b +w_1x_1 + ... +w_dx_d$$로 볼 수 있다.
    편향 $$b$$는 아무런 feature들을 관찰 하지 않고 한 예측이다.
    모델의 무게 $$w_i$$는 feature와 어떻게 양의 label과 상관관계를 가지는지 나타낸다.
    만약에 $$x_i$$가 양수 label과 양의 상관관계를 가지면 무게 $$w_i$$는 증가할 것이고 $$P(Y=1|\mathbf{x})$$는 1에 가까울 것이다. 반면에, $$x_i$$이 양수 label과 음의 상관관계를 가지면 무게 $$w_i$$는 감소 할 것이고 $$P(Y=1|\mathbf{x})$$는 0에 가까울 것이다.

*   **로지스틱 함수**: 두 번째로, 우리는 로지스틱 함수(시그모이드 함수로 알려진) $$S(t) = 1/(1+\exp(-t))$$가 선형 모델에 적용되는 것을 볼 수 있다. 확률로 해석되어질 수 있는 어떠한 0 과 1 사이의 실수로에서 나온 선형모델 $$\mathbf{w}^T\mathbf{x}+b$$의 출력 값을 변환 하는데 사용된다.

모델 훈련은 최적화에 대한 문제이다. 다시말해, 데이터를 학습하는 동안 정의된 **손실 함수**를 최소화 할수 있는 모델 weights 세트(즉, 모델 매개변수)를 찾는 것이다. 손실 함수는 라벨의 실측 라벨과 모델의 예측 차이점을 측정한다. 만약에 예측이 실수 라벨과 아주 가깝다면 손실 값은 낮을 것이고 반대로 예측값이 라벨과 많은 차이가 날 때 손실 값은 높을 것이다.

## 심화학습

만약 더 많은 것을 배우고 싶다면 어떻게 선형 모델과 심층 신경망에 강점을 TF.Learn API을 이용하여 훈련 시키는지 알아 볼 수 있는 [Wide & Deep Learning Tutorial](../wide_and_deep/)을 확인해 보시오.
