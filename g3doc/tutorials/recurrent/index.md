# 순환 신경망(Recurrent Neural Networks)
(v0.9)

## 소개

순환 신경망과 LSTM에 관한 소개는 이
[블로그](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)를 참고하세요.

## 언어 모델링(Language Modeling)

이 튜토리얼에서 언어 모델링 문제에 대해 순환 신경망을 어떻게 학습시키는지 살펴 보겠습니다.
여기서 풀려고 하는 문제는 문장 구성에 대한 확률을 부여하는 모델을 최적화시키는 것입니다.
즉 이전에 나타난 단어의 기록을 보고 다음의 단어를 예측하는 것입니다.
우리가 사용할 데이터는 이런 종류의 모델의 성능을 평가하는 데 널리 사용되고
비교적 크기가 작아 학습하는 데 시간이 많이 걸리지 않는
[Penn Tree Bank](http://www.cis.upenn.edu/~treebank/) (PTB) 데이터셋입니다.

언어 모델링은 음성 인식, 기계 번역, 이미지 캡셔닝(captioning) 같이 인기있는 여러 분야의 핵심 요소입니다.
또한 매우 재미있습니다.
[여기](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)를 한번 둘러보세요.

이 튜토리얼에서는 PTB 데이터셋을 사용하여 뛰어난 성과를 낸
[Zaremba 외., 2014](http://arxiv.org/abs/1409.2329)
([pdf](http://arxiv.org/pdf/1409.2329.pdf))의 결과를 재현할 것 입니다.

## 튜토리얼 파일

이 튜토리얼에서 사용할 파일들은 `models/rnn/ptb`에 있습니다.

파일 | 설명
--- | ---
`ptb_word_lm.py` | 이 코드는 PTB 데이터셋을 사용하여 언어 모델을 학습시킵니다.
`reader.py` | 이 코드는 데이터를 읽어 들이는데 사용됩니다.

## 데이터 다운로드하여 준비하기

이 튜토리얼에서 필요한 데이터는 Tomas Mikolov의 웹 페이지에서 다운 받은 파일의 data 디렉토리
안에 있습니다: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

이 데이터셋은 전처리된 총 만개의 단어를 가지고 있고 문장 끝을 나타내는 표시와 희귀 단어를 표시하는
심볼(&lt;unk&gt;)을 포함하고 있습니다. 이 데이터를 신경망에서 처리하기 적합하도록
`reader.py` 에서 모든 단어를 고유한 정수 숫자로 바꿉니다.

## 학습 모델

### LSTM

학습 모델의 핵심 부분은 한번에 하나의 단어를 입력받아 문장에서 나타날 다음 단어의 확률을
계산하는 LSTM 셀(cell)로 이루어져 있습니다. 신경망의 메모리 상태는 0으로 초기화되고 단어를
읽으면서 업데이트 됩니다. 계산 효율을 높이기 위해 `batch_size` 크기의 미니배치(mini-batch)로
모델을 학습시킬 것입니다.

기본적인 의사코드는 아래와 같습니다:

```python
lstm = rnn_cell.BasicLSTMCell(lstm_size)
# LSTM 상태 메모리 초기화.
state = tf.zeros([batch_size, lstm.state_size])

loss = 0.0
for current_batch_of_words in words_in_dataset:
    # 상태 값은 배치를 처리한 후 업데이트 됩니다.
    output, state = lstm(current_batch_of_words, state)

    # LSTM의 출력 값을 사용하여 다음 단어를 예측합니다.
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, target_words)
```

### 부분 역전파(Truncated Backpropagation)

학습 과정을 지켜보기 위해서 일정 횟수(`num_steps`)만큼 학습을 진행한 후에 그 만큼의
그래디언트만 역전파 시키는 것이 보통입니다. 반복 루프안에서 한번에 `num_steps` 길이 만큼
입력 값을 주입하고 나서 역전파 시키면 됩니다.

부분 역전파를 위한 그래프를 만드는 코드의 간소화 버전은 아래와 같습니다:

```python
# 한번 반복에서 처리할 입력 값을 위한 플레이스홀더
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = rnn_cell.BasicLSTMCell(lstm_size)
# LSTM 상태 메모리 초기화.
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(num_steps):
    # 상태 값은 배치를 처리한 후 업데이트 됩니다.
    output, state = lstm(words[:, i], state)

    # 이어진 코드
    # ...

final_state = state
```

그리고 아래는 전체 데이터셋에 대해 반복하는 부분을 구현한 것입니다:

```python
# 배치를 처리한 후 LSTM의 상태를 저장하는 numpy 배열.
numpy_state = initial_state.eval()
total_loss = 0.0
for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = session.run([final_state, loss],
        # 이전 반복에서 얻은 결과를 사용해 LSTM 상태를 초기화.
        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss
```

### 입력

단어의 아이디(ID)는 LSTM에 주입되기 전에 밀집 행렬(dense representation)([벡터 표현 튜토리얼](../../tutorials/word2vec/index.md)을 참고하세요)에 임베딩(embedding)될 것입니다.
이 방식은 특정 단어에 대한 정보를 효과적으로 표현할 수 있습니다. 아래와 같이 만들 수 있습니다:

```python
# embedding_matrix는 [단어수, 임베딩사이즈] 크기의 텐서입니다.
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
```

임베딩 행렬은 랜덤하게 초기화되고 데이터를 처리하면서 단어의 의미를 구분하도록 학습됩니다.

### 손실 함수(Loss Function)

우리는 목적 단어의 로그 확률의 음수 평균을 최소화하려고 합니다:

$$ \text{loss} = -\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i} $$

직접 구현하는 것도 어렵지 않으나 이미 `sequence_loss_by_example` 함수가 있어 이를 사용하겠습니다.

이 페이퍼에서 사용한 측정법은 평균 단어당 복잡도(perplexity)입니다(종종 그냥 복잡도라고 부릅니다).
아래 식과 같습니다.

$$e^{-\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i}} = e^{\text{loss}} $$

학습 과정 동안 이 값을 모니터링하도록 하겠습니다.

### LSTM 레이어 만들기

모델의 성능을 높이기 위해 데이터를 처리할 여러개의 LSTM 레이어를 만들 수 있습니다.
첫번째 레이어의 출력은 두번째 레이어의 입력이 되는 식입니다.

이런 작업을 위해 구현된 `MultiRNNCell` 클래스가 있습니다:

```python
lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(num_steps):
    # 상태 값은 배치를 처리한 후 업데이트 됩니다.
    output, state = stacked_lstm(words[:, i], state)

    # 이어진 코드
    # ...

final_state = state
```

## 코드 실행

독자가 이미 pip 패키지를 통해 텐서플로우를 설치했고 텐서플로우 깃 저장소(git repository)에서
클론하여 깃 트리의 최상위 디렉토리에 있다고 가정합니다. (만약 [소스에서 직접 빌드](
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#installing-from-sources) 했다면 [bazel](https://github.com/bazelbuild/bazel)을 이용해서
`tensorflow/models/rnn/ptb:ptb_word_lm` 타겟을 빌드하세요).

그 다음은:
```bash
cd tensorflow/models/rnn/ptb
python ptb_word_lm --data_path=/tmp/simple-examples/data/ --model small
```

이 튜토리얼에 포함된 코드에는 세가지의 모델 환경을 제공합니다: "small", "medium", "large" 입니다.
이들간의 차이는 LSTM 레이어의 크기와 학습에 사용될 하이퍼파라미터(hyperparameter) 설정입니다.

큰 모델일수록 더 좋은 결과가 나와야 합니다. 'small' 모델은 테스트 셋에 대한 복잡도가 120 아래에 도달하며
'large' 모델은 80 이하가 나오지만 학습에 여러시간이 소요될 수 있습니다.

## 그 다음엔?

여기서 언급하지 않았지만 모델의 성능을 더 좋게 만들기 위한 몇가지 기법이 있습니다:

* 학습 속도 감소를 스케줄링 하기,
* LSTM 레이어 간에 드롭아웃 적용.

코드를 연구하고 수정해서 모델의 성능을 개선해 보십시요.
