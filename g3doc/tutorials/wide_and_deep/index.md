# TensorFlow Wide & Deep Learning 튜토리얼

이전 튜토리얼[선형모델(Linear Model) 튜토리얼](../wide/) 에서, [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) 을 이용하여 연소득이 $50,000 이상인 사람을 예측하기 위한 로지스틱스 회귀모델을 학습했다. TensorFlow 는 deep neural networks(신경망) 학습에 탁월함은 물론, 어느 것을 잘 선택해야 하는가에 대한 것을 생각할 수 있다. 이들을 동시에는 안될까? 하나의 모델에 둘의 strength를 조합하는 것은 가능할까? 

이 튜토리얼에서, TF 를 사용하는 법에 대해 소개할 것이다. wide 선형 모델과 deep feed-forward 신경망을 함께 학습하기 위한 API 를 배우자. 이 접근법은 저장법과 일반화의 강점을 조합한 것이다. 이 방법은 sparse 입력 feature 들(즉, 많은 수의 가능한 특징값을 가진 분류적 특징) 의 일반적으로 큰 규모의 회귀법과 분류법 문제들에 유용하다. Wide & Deep 학습의 학습 동작에 대해 관심이 있다면, 우리 논문[research paper](http://arxiv.org/abs/1606.07792) 를 참고하기 바란다.

![Wide & Deep Spectrum of Models]
(../../images/wide_n_deep.svg "Wide & Deep")

위 그림은 wide 모델(sparse feature 와 변환법의 로지스틱 회귀분석), deep 모델(embedding layer(층)와 여러 hidden layer(층)들의 feed-forward 신경망) 을 비교하여 보여준다. 가장 높은 수준에서는, TF를 이용한 wide, deep, wide & deep 모델의 설정 3 단계만 존재한다. API 를 배워보자:

1. wide 부분에 대한 features 선택 : 사용하길 원하는 sparse base column 과 crossed column 들을 선택한다
1. deep 부분에 대한 features 선택 : 연속된 열, 각 분류 열의 embedding dimension, 그리고 hidden layer 크기를 선택한다
1. 이들을 Wide & Deep 모델에 적용한다(`DNNLinearCombinedClassifier`)

다 됐다! 간단한 예제를 통해 알아보자.

## 설정

이 튜토리얼을 위한 코드를 실행하기 위해서,

1.  아직 설치되지 않았다면, TensorFlow를 설치한다 [Install TensorFlow](../../get_started/os_setup.md)

2.  튜토리얼 코드를 다운로드 한다 [the tutorial code](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)

3.  pandas 데이터 분석 라이브러리를 설치한다. tf.learn은 pandas 을 필요로 하지 않지만 이를 지원한다. 그리고 이 튜토리얼은 pandas를 사용한다. pandad 를 설치하기 위해서 :
	1. Get `pip`:

       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```


       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```


       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```

    2. Use `pip` to install pandas:
	2. `pip` 를 이용하여 pandas 설치:

       ```shell
       $ sudo pip install pandas
       ```

	pandas 설치에 문제가 있다면, pandas 사이트의 [instructions](http://pandas.pydata.org/pandas-docs/stable/install.html) 를 찾아보아라.

4. 이 튜토리얼에 소개된 선형 모델을 학습시키기 위해 이어지는 명령어로 튜토리얼 코드를 실행하자:

   ```shell
   $ python wide_n_deep_tutorial.py --model_type=wide_n_deep
   ```

이 코드가 어떻게 이 선형 모델을 만들어가는 알아보기 위해 계속 읽어보자.


## Base Feature Columns 정의

우선, 사용할 base categorical feature column 과 continuous feature column 들을 정의하자. 이들 base column 들은 모델의 wide 부분과 deep 부분에 모두 사용될 building block 들이 된다.

```python
import tensorflow as tf

# Categorical base columns.
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female", "male"])
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Continuous base columns.
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
```

## The Wide Model: Linear Model with Crossed Feature Columns

wide 모델은 sparse 하고 crosseed feature column 들의 다양한 집합의 선형 모델이다:

```python
wide_columns = [
  gender, native_country, education, occupation, workclass, marital_status, relationship, age_buckets,
  tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([age_buckets, race, occupation], hash_bucket_size=int(1e6))]
```

crossed feature column 들의 Wide 모델은 feature들 간의 sparse interactions(상호작용들)을 효율적으로 저장할 수 있다. 그럼에도, crossed feature columns 의 한가지 제한되는 점은 학습된 데이터에 나타나지 않는 feature 조합들을 생성해내지 못한다는 것이다. 이를 수정하기 위해 embeddings 를 포함한 deep 모델을 추가해보자.

## The Deep Model: Neural Network with Embeddings

이전 그림에서 보이는 것과 같이 deep 모델은 feed-forward 신경망이다. 각각의 sparse, high-dimensional(고차원의) categorical feature 들은 종종 embedding 벡터라 불리는 low-dimensional(저차원의), dense real-valued 벡터로 변환된다. 이 low-dimensional(저차원의) dense embedding 벡터들은 continuouse feature 들로 연계되어 변환되고, 계속해서 다음 신경망의 hidden layers 로 연결된다. embedding value 들은 무작위로 초기화되고, 학습 loss(손실)을 최소화하는 다른 모든 모델 파라미터들과 함께 학습된다. embeddings 에 대한 학습에 흥미가 있다면 TensorFlow 튜토리얼 [Vector Representations of Words](https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html) 이나 위키피디아 [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding) 를 확인해보자.

`embedding_column` 을 이용하여 categorical column 들을 위한 embeddings 를 설정할 것이다. 그리고 이들을 continuous columns 과 연관지을 것이다.

```python
deep_columns = [
  tf.contrib.layers.embedding_column(workclass, dimension=8),
  tf.contrib.layers.embedding_column(education, dimension=8),
  tf.contrib.layers.embedding_column(marital_status, dimension=8),
  tf.contrib.layers.embedding_column(gender, dimension=8),
  tf.contrib.layers.embedding_column(relationship, dimension=8),
  tf.contrib.layers.embedding_column(race, dimension=8),
  tf.contrib.layers.embedding_column(native_country, dimension=8),
  tf.contrib.layers.embedding_column(occupation, dimension=8),
  age, education_num, capital_gain, capital_loss, hours_per_week]
```

embedding 의 더 높은 `dimension` 은 더 높은 자유도로 features 의 대표들을 학습할 수 있다. 간단하게, 여기 모든 feature columns 의 dimension(차원)을 8 로 정했다. 경험적으로, a more informed decision 차원의 수는 $$k\log_2(n)$$ 나 $$k\sqrt[4]n$$ 의 차수 값에서 시작한다. 이때 $$n$$ 은 feature column 에서 유니크한 feature 의 수이고 $$k$$ 는 작은 상수값(일반적으로 10보다 작은 값)이다.

dense embeddings 를 통해, deep 모델은 학습된 데이터에서 이전에 보여지지 못한 feature 쌍들에 대한 생성을 더 잘 할 수 있고 예측할 수 있다. 그러나 두 feature column 간의 기본 interaction matrix(상호작용 메트릭스)가 sparse 하고 high-rank 일 때, feature column 들에 대한 low-dimensional representation 들의 효율적인 학습은 어렵다. 이러한 경우, 대부분의 feature 쌍 들간 interaction(상호작용)은 몇몇을 제외하는 0(zero) 이 되어야 한다. 하지만 dense embeddings 는 모든 feature 쌍들에 대해 none-zero(0 이 아닌) 예측을 이끌어 냄에 따라, 지나치게 일반화할 수 있다. 반면, crossed features 선형모델은 이러한 "exception rules"(예외들)을 더 적은 모델 파라미터들을 이용해 효율적으로 저장할 수 있다.

이제 어떻게 wide 모델과 deep모델을 함께 학습시키고 그들 각각의 강점과 약점을 보완해 가는지 살펴보자.

## Wide and Deep 모델을 하나로 조합하기

wide 모델과 deep 모델들은 예측으로써 그들의 마지막 출력 로그 나머지(final output log odd) 의 합에 의해 조합된다. 그 다음 로지스틱 loss 함수의 예측에 전달된다. 모든 그래프 정의와 변수 할당량들은 이미 내부에서 다뤄졌기 때문에, `DNNLinearCombinedClassifier` 를 만들어 볼 필요가 있다.:

```python
import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```

## 모델 학습하기와 평가하기

모델 학습시키기 전에, 우리가 [TensorFlow Linear Model tutorial](../wide/) 에서 했던 인구조사 데이터 세트를 입력하자. 입력 데이터 처리를 위한 코드는 편의를 위해 다시 제공한다:

```python
import pandas as pd
import urllib

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

# Download the training and test data to temporary files.
# Alternatively, you can download them yourself and change train_file and
# test_file to your own paths.
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

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

데이터를 읽고 난 후, 모델을 학습하고 평가할 수 있다:

```python
m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
  print "%s: %s" % (key, results[key])
```

결과의 첫줄에는 `정확도: 0.84429705` 와 같이 나타나야 한다. wide 만 사용한 선형모델에서 83.6% 인 정확도가 Wide & Deep 모델을 사용한 것에서 84.4% 로 향상된 것을 볼 수 있다. 완전한 예제를 경험하길 원한다면, [example code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py) 를 다운로드 할 수 있다.

이 튜토리얼은 API 에 친숙하게 하기 위한 작은 데이터세트(dataset)의 간단한 예제임을 알아두자. Wide & Deep 러닝은 가능한 많은 수의 feature value 값들을 가지는 많은 sparse feature column 들의 데이터세트(dataset) 로 시도할 경우 더욱 강력할 것이다. 다시말해, Wide & Deep 러닝을 대규모의 실세계에 맞춰진 러닝 문제들을 어떻게 적용하는지에 대한 많은 아이디어를 위해 우리의 [research paper](http://arxiv.org/abs/1606.07792) 을 자유롭게 참고하라.
