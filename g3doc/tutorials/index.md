# 튜토리얼

## 기본 뉴럴 네트워크

### MNIST 초급

만약 기계학습을 처음 배우신다면, 여기서부터 출발하기를 권장드립니다. MNIST라는 손으로 쓰여진 숫자를 분류하는 전통적인 문제에 대해 배워, 다중 분류에 관한 가벼운 입문을 할 수 있습니다.

[튜토리얼 보기](../tutorials/mnist/beginners/index.md)


### MNIST 고급

만약 이미 다른 딥러닝 소프트웨어 패키지와 MNIST에 익숙하시다면, 이 튜토리얼이 텐서플로우에 관한 매우 간단한 시작점이 될 것입니다.

[튜토리얼 보기](../tutorials/mnist/pros/index.md)

### 텐서플로우 구조

규모 있는 모델을 학습시키기 위해 텐서플로우의 인프라를 조금 더 자세하게 알아보고자 할 때의 기술적인 튜토리얼입니다. MNIST를 예시로 이용합니다.

[튜토리얼 보기](../tutorials/mnist/tf/index.md)

### MNIST 데이터 다운로드

MNIST 숫자 데이터셋을 다운로드하는 것에 대한 자세한 사항입니다. 꽤 흥미로운 자료입니다.

[튜토리얼 보기](../tutorials/mnist/download/index.md)


## tf.contrib.learn을 사용한 간편한 머신러닝

### tf.contrib.learn 시작하기

텐서플로우의 고수준 API인 tf.contrib.learn의 간략한 입문입니다. 단 몇 줄의 코드로 신경망을 만들고, 훈련하고, 평가합니다.

[튜토리얼 보기](../tutorials/tflearn/index.md)

### tf.contrib.learn 선형모델 소개

텐서플로우에서 선형 모델을 작업하기 위해서, tf.contrib.learn의 풍부한 도구에 대해 소개합니다.

[튜토리얼 보기](../tutorials/linear/overview.md)

### 선형모델 튜토리얼

이 튜토리얼은 tf.contrib.learn을 이용해 선형 모델을 만드는 코드를 살펴볼 수 있습니다.

[튜토리얼 보기](../tutorials/wide/index.md)

### 와이드앤 딥러닝 튜토리얼

이 튜토리얼은 각 모델의 장점을 결합하기 위해 선형 모델과 딥 신경망을 tf.contrib.learn을 이용해 동시에 학습시키는 방법을 보여줍니다.

[튜토리얼 보기](../tutorials/wide_and_deep/index.md)


## 텐서플로우 서빙

### 텐서플로우 서빙

생산적인 환경 마련을 위한, 기계학습 모델을 제공하는 유연하고 우수한 시스템인 텐서플로우 서빙을 소개합니다.

[튜토리얼 보기](../tutorials/tfserve/index.md)


## 이미지 프로세싱

### 콘볼루션 뉴럴 네트워크

CIFAR-10 데이터셋을 이용한 콘볼루션 신경망에 관한 소개입니다. 시각적인 자료에 관해 더 함축적이고 효과적인 표현을 산출하기 위해 변환 불변성을 이용하기 때문에, 콘볼루션 신경망은 특별히 이미지 처리에 맞게 설계되어 있습니다.

[튜토리얼 보기](../tutorials/deep_cnn/index.md)

### 이미지 인식

ImageNet Challenge 데이터와 라벨 데이터셋으로 훈련된 콘볼루션 신경망을 이용해 물체를 인식하는 법을 배웁니다.

[튜토리얼 보기](../tutorials/image_recognition/index.md)

### Deep Dream 시각 환상

인셉션 인식 모델을 구축한, [Deep Dream](https://github.com/google/deepdream) 신경망 시각 환상 소프트웨어의 텐서플로우 버전을 배포합니다.

[튜토리얼 보기](https://www.tensorflow.org/code/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)


## 언어와 시퀀스 프로세싱

### word2vec 모델

이 튜토리얼은 단어를 벡터로 표현하는 방법(*워드 임베딩*)을 배우는 게 어째서 유용한지에 관해 여러분의 흥미를 자극할 것입니다. 임베딩을 학습하기 위한 효과적인 모델로 word2vec 모델을 소개합니다. 또한, 노이즈-대조 학습 방법을 지지하는 높은 수준의 디테일도 다룰 것입니다.

[튜토리얼 보기](../tutorials/word2vec/index.md)

### 리커런트 뉴럴 네트워크

영어 문장에서 다음 단어를 예측하기 위해 LSTM 네트워크를 학습하는 RNN에 관한 소개입니다. (언어 모델링이라고도 부르는 작업입니다.)

[튜토리얼 보기](../tutorials/recurrent/index.md)

### seq2seq 모델

RNN 튜토리얼에 이어, 기계 번역을 위해 시퀀스-시퀀스 모델을 결합합니다. 전체적으로 머신이 학습해, 자신 만의 영어-프랑스어 번역기를 만드는 것을 배웁니다.

[튜토리얼 보기](../tutorials/seq2seq/index.md)

### SyntaxNet

텐서플로우를 위한 자연언어처리 프레임워크인 SyntaxNet을 소개합니다.

[튜토리얼 보기](../tutorials/syntaxnet/index.md)


## 비머신러닝 애플리케이션

### 만델브로트

텐서플로우는 기계학습과 관련이 없는 연산 작업에도 활용될 수 있습니다. 여기서는 만델브로트 데이터셋을 시각화하는 나이브한 구현을 보여줍니다.

[튜토리얼 보기](../tutorials/mandelbrot/index.md)

### 편미분 방정식

기계학습과 관련이 없는 또 다른 예시로, 연못에 떨어지는 빗방울에 관한 나이브한 PDF 시뮬레이션의 예시를 제공합니다.

[튜토리얼 보기](../tutorials/pdes/index.md)
