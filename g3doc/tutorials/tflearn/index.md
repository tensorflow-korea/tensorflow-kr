<!-- 160708 @mikibear : 최종 배포시에 md로 링크된 것을 html로 변경할 것입니다. 혹시 발견하신다면, 변경해주시면 감사하겠습니다. -->

## tf.contrib.learn 시작하기

텐서플로우의 고수준 머신러닝 API(tf.contrib.learn)는 다양한 머신러닝 모델을 쉽게 설정하고, 훈련하고, 평가할 수 있도록 해줍니다. 이 간단한 튜토리얼에서는  [신경망](https://en.wikipedia.org/wiki/Artificial_neural_network) 분류기를 만들고, [피셔의 Iris 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)으로 꽃받침과 꽃잎의 정보를 이용하여 꽃의 종류를 예측할 수 있도록 분류기를 훈련시킬 것입니다. 당신은 다음의 다섯 단계를 수행할 것입니다:

1. Iris 훈련/테스트 데이터를 담은 CSV 파일을 텐서플로우 Dataset으로 불러옵니다
2. [신경망 분류기](../../api_docs/python/contrib.learn.md#DNNClassifier)를 만듭니다
3. 훈련 데이터를 이용하여 모델을 피팅합니다
4. 모델의 정확도를 평가합니다
5. 새로운 표본을 분류합니다

## 시작하기

이 튜토리얼을 시작하기 전에  [당신의 머신에는 텐서플로우가 설치되어 있어야합니다](../../get_started/os_setup.md).

다음은 우리의 신경망의 전체 코드입니다 :

```python
import tensorflow as tf
import numpy as np

# 데이터셋
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 데이터셋을 불러옵니다.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target

# 10-20-10의 구조를 갖는 3층 DNN를 만듭니다
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

# 모델을 피팅합니다.
classifier.fit(x=x_train, y=y_train, steps=200)

# 정확도를 평가합니다.
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# 새로운 두 꽃의 표본을 분류합니다.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))
```

다음의 문단들은 이 코드를 자세하게 살펴볼 것입니다.

## Iris 데이터 CSV 파일을 텐서플로우로 불러오기

[Iris 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)은 Iris 종 내에서 서로 연관된 *Iris setosa*, *Iris virginica*, and *Iris versicolor* 세 종이 각각 50개씩의 표본으로 구성되어 150개 행의 데이터로 되어있습니다. 각각의 행은 각 꽃의 표본에 대한 다음의 정보를 담고 있습니다 : [꽃받침](https://en.wikipedia.org/wiki/Sepal) 길이, 꽃받침 너비, [꽃잎](https://en.wikipedia.org/wiki/Petal) 길이, 꽃잎 너비, 그리고 꽃의 종류. 꽃의 종류는 정수로 표현되어 있으며, 0은 *Iris setosa*, 1은 *Iris versicolor*, 그리고 2는 *Iris virginica*를 나타냅니다.

꽃받침 길이 | 꽃받침 너비 | 꽃잎 길이 | 꽃잎 너비 | 종
:----------- | :---------- | :----------- | :---------- | :------
5.1          | 3.5         | 1.4          | 0.2         | 0
4.9          | 3.0         | 1.4          | 0.2         | 0
4.7          | 3.2         | 1.3          | 0.2         | 0
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
7.0          | 3.2         | 4.7          | 1.4         | 1
6.4          | 3.2         | 4.5          | 1.5         | 1
6.9          | 3.1         | 4.9          | 1.5         | 1
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
6.5          | 3.0         | 5.2          | 2.0         | 2
6.2          | 3.4         | 5.4          | 2.3         | 2
5.9          | 3.0         | 5.1          | 1.8         | 2

<!-- 유의사항 : 이 문단의 나머지에서는 CSV 파일이 튜토리얼 예제 파일과 같은 경로에 있다고 가정합니다 : 만약 그렇지 않다면, 링크와 코드를 갱신하십시오. -->
이 튜토리얼을 위해서는 Iris data는 임의적으로 섞인 후에, 두 개의 따로 떨어진 CSV 파일로 나누어져야 합니다. 이는 120개의 표본을 갖는 훈련 데이터([iris_training.csv](http://download.tensorflow.org/data/iris_training.csv))와 30개의 표본을 갖는 테스트 데이터([iris_test.csv](http://download.tensorflow.org/data/iris_test.csv))입니다.

시작하기 위해선, 먼저 텐서플로우와 numpy를 불러옵니다 : 

```python
import tensorflow as tf
import numpy as np
```

그 다음, `learn.datasets.base`에 있는 [`load_csv()`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/datasets/base.py) 함수를 이용하여 훈련 셋과 테스트 셋을 `Dataset`으로 불러옵니다. `load_csv()` 함수는 두 개의 인자를 요구합니다.

*   `filename`, CSV 파일이 존재하는 파일의 경로
*   `target_dtype`, dataset의 목표 값의 [`numpy` 데이터형](http://docs.scipy.org/doc/numpy/user/basics.types.md)

여기에서 목표 값(모델을 훈련시켜 예측하려고 하는 값)은 0&ndash;2의 정수로 구성된 꽃의 종입니다. 따라서, 적절한 `numpy` 데이터형은 `np.int`입니다.

```python
# 데이터셋
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 데이터셋을 불러옵니다.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)
```

그 다음, 길이 및 너비 등의 특성 데이터와 목표 값에 변수를 할당합니다. 훈련 데이터셋의 특성 데이터는 `x_train`, 테스트 데이터셋의 특성 데이터는 `x_test`, 훈련 데이터셋의 목표 값은 `y_train`, 테스트 데이터셋의 목표 값은 `y_test`입니다. tf.contrib.learn의 `Dataset`은 [named tuples](https://docs.python.org/2/library/collections.html#collections.namedtuple)이며, 순차적으로 데이터와 목표 필드*(역주 : namedtuple의 field_name을 말합니다)*를 통해 특성 데이터와 목표 값에 접근할 수 있습니다.

```python
x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
```

후술할 “Iris 훈련 데이터로 DNNClassifier 피팅하기”에서, `x_train`과 `y_train`을 이용하여 모델을 훈련시키고, “모델 정확도 평가하기”에서는 `x_test`와 `y_test`를 이용할 것입니다. 하지만 먼저, 다음 문단에선 모델을 구성해봅시다.

## 딥 인공신경망 분류기 만들기

tf.contrib.learn은 데이터로 훈련과 평가를 실행할 수 있도록 곧장 사용할 수 있는, [`Estimator`](../../api_docs/python/contrib.learn.md#estimators)라 불리는 여러 가지의 미리 정의된 모델을 제공합니다. 여기에서는 Iris data를 피팅하기 위해 딥 인공 신경망 모델을 설정하도록 합시다. tf.contrib.learn을 이용하면, [`DNNClassifier`](../../api_docs/python/contrib.learn.md#DNNClassifier)를 한 줄 만에 인스턴스화할 수 있습니다.

```python
# 10-20-10의 구조를 갖는 3층 DNN를 만듭니다
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
```

위의 코드는 각각 10, 20, 10개의 뉴런으로 이루어진 3개의 [은닉층](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)을 포함한 `DNNClassifier` 모델을 생성합니다. 이는 (`hidden_units=[10, 20, 10]`), 그리고 세 개의 목표 분류 (`n_classes=3`)에 순차적으로 따른 것입니다.

## Iris 훈련 데이터로 DNNClassifier 피팅하기

이제 DNN `classifier` 모델을 설정했으니, [`fit`](../../api_docs/python/contrib.learn.md#BaseEstimator.fit) 메소드를 이용하여 Iris 훈련 데이터로 이를 피팅할 수 있습니다. 특성 데이터(`x_train`)와 목표 값(`y_train`), 그리고 train할 단계 수(여기서는 200) 인자로 넘겨줍니다.

```python
# Fit model
classifier.fit(x=x_train, y=y_train, steps=200)
```

<!-- Style the below (up to the next section) as an aside (note?) -->

<!-- Pretty sure the following is correct, but maybe a SWE could verify? -->

`classifier`에서 모델의 상태는 유지됩니다. 이는, 만약 원한다면 모델을 반복하여 학습시킬 수 있다는 것을 의미합니다. 예를 들어서, 위의 한 줄은 다음의 두 줄과 완벽하게 같습니다.

```python
classifier.fit(x=x_train, y=y_train, steps=100)
classifier.fit(x=x_train, y=y_train, steps=100)
```

<!-- TODO: When tutorial exists for monitoring, link to it here -->
하지만, 만약 학습되는 동안에 모델을 추적하고 싶은 것이라면, (위와 같은 두 줄) 대신에 로그를 남기기 위해서 텐서플로우의 [`monitor`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/monitors.py)를 사용하는 게 낫습니다.

## 모델 정확도 평가하기

이제 Iris 테스트 데이터에 맞춰 `DNNClassifier` 모델을 피팅했습니다. 이제, [`evaluate`](../../api_docs/python/contrib.learn.md#BaseEstimator.evaluate) 메소드를 이용하여 Iris 테스트 데이터로 모델의 정확도를 확인해볼 수 있습니다. `evaluate`는 `fit`과 같이 특성 데이터와 목표 값을 인자로 건내받고, 평가 결과로서 `dict`를 반환합니다. 다음의 코드는 Iris 테스트 데이터&mdash;`x_test`와 `y_test`&mdash;를 건내받아, 결과값으로 `accuracy`를 출력합니다.

```python
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
```

전체 스크립트를 실행하고 나서 정확도의 결과를 확인합시다. 다음과 같은 결과를 얻을 수 있을 것입니다:

```
Accuracy: 0.933333
```

비교적 적은 데이터셋 치고는 나쁘지 않습니다!

## 새로운 표본 분류하기

새로운 표본을 분류하기 위해 estimator의 `predict()` 메소드를 이용합시다. 예를 들어, 다음의 두 가지 새로운 꽃의 표본이 있다고 해봅시다 : 

꽃받침 길이 | 꽃받침 너비 | 꽃잎 길이 | 꽃잎 너비
:----------- | :---------- | :----------- | :----------
6.4          | 3.2         | 4.5          | 1.5
5.8          | 3.1         | 5.0          | 1.7        

다음의 코드로 이들의 종을 예측할 수 있습니다 : 

```python
# 새로운 두 꽃의 표본을 분류합니다.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))
```

`predict()` 메소드는 각 표본의 예측 결과를 하나씩 배열로 반환합니다.

```python
Prediction: [1 2]
```

따라서 모델은 첫 번째 표본을 *Iris versicolor*, 두 번째 표본을 *Iris virginica*로 예측하였습니다.

## 추가적인 자료

* tf.contrib.learn에 대해 추가적인 참고 자료를 원한다면, 공식적인 [API docs](../../api_docs/python/contrib.learn.md)를 살펴보십시오.

<!-- David, will the below be live when this tutorial is released? -->
* 선형 모델을 생성하기 위해서 tf.contrib.learn을 이용하는 것에 대해 좀 더 배우기 위해선 [Large-scale Linear Models with TensorFlow](../linear/)를 살펴보십시오.

* 브라우저에서의 신경망 모델링과 시각화를 체험해보기 위해선, [Deep Playground](http://playground.tensorflow.org/)를 살펴보십시오.

* 신경망에 대한 좀 더 심화된 튜토리얼을 원한다면 [Convolutional Neural Networks](../deep_cnn/)와 [Recurrent Neural Networks](../recurrent/)를 살펴보십시오.
