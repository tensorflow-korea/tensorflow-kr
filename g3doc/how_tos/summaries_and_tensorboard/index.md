# TensorBoard: 학습 시각화
(v0.12)

TensorFlow를 쓸려는 연산은 거대한 심층 신경망 학습처럼 복잡하고 내용이 헷갈리는 것들입니다. 이해, 디버깅, 또 TensorFlow 최적화를 쉽게 만들기 위해서 시각화 도구 세트를 하나 넣어뒀습니다. 바로 TensorBoard입니다. 여러분은 TensorBoard를 이용해서 TensorFlow의 그래프를 시각화하고 그래프를 실행해서 얻은 행렬을 도표로 나타내고 이미지 파일같은 부가 데이터를 보여줄 수도 있습니다. TensorBoard가 완전히 셋팅되면 이렇게 보일 것입니다:

![MNIST TensorBoard](../../images/mnist_tensorboard.png "MNIST TensorBoard")


이 튜토리얼은 간단한 TensorBoard 사용법을 안내하는 것이 목적입니다. 참고할 만한 다른 문서들도 있어요! [TensorBoard 길라잡이](https://www.tensorflow.org/code/tensorflow/tensorboard/README.md)는 TensorBoard 사용법에 대한 팁이나 디버깅 방법 등 더 많은 정보를 담고 있습니다.

## 데이터 저장(serialize)

TensorBoard는 TensorFlow를 실행할 때 만들 수 있는 요약 데이터(summary data)가 들어간 TensorFlow 이벤트 파일을 이용합니다. TensorBoard에서 요약 데이터가 보여주는 라이프 사이클을 설명해드리겠습니다.

가장 먼저 요약 데이터를 얻고 싶은 TensorFlow 그래프를 만들어야겠죠. 그리고 [summary operations](../../api_docs/python/train.md#summary-operations)을 이용해서 어느 노드를 기록할 지 결정합니다.

예를 들어 지금 곱집합 신경망을 학습시켜서 MNIST 숫자들(역자 주: 프로그래밍의 'hello world'처럼 이미지 인식에서 가장 기초적인 예제)을 인식하려고 한다고 해봅시다. 아마도 학습률이 어떻게 달라지는지, 목표함수가 어떻게 바뀌는지를 기록하고 싶을 것입니다. 학습률과 손실을 각각 만들어내는 노드에 [`scalar_summary`](../../api_docs/python/train.md#scalar_summary) 작업(op)을 추가해서 데이터를 모을 수 있습니다. 그리고 각 `scalar_summary`에는 `학습률`이나 `손실함수`같은 `태그`를 붙일 수도 있습니다.

특정 층에서 발생한 액티베이션, 그래디언트 혹은 가중치의 분포를 시각화하고 싶을 것 같기도 합니다. 이럴 때는 그래디언트 결과물이나 가중치 변수에 [`histogram_summary`](../../api_docs/python/train.md#histogram_summary) 작업(op)을 추가해서 데이터를 모을 수 있습니다.

가능한 모든 요약 작업(summary operation)들은 [summary operations](../../api_docs/python/train.md#summary-operations) 문서를 확인하시기 바랍니다.

TensorFlow의 작업(op)들은 이용자가 그 작업이나 연관된 다른 작업을 실행시킬 때까지 아무 것도 하지 않습니다. 우리가 만든 요약 노드들은 그래프에서 지엽적인 존재입니다. 아무 노드도 요약 노드들의 결과에 영향을 받지 않거든요. 그렇기 때문에 요약 데이터를 만들려면 반드시 모든 요약 노드들을 실행시켜야 합니다. 일일이 손으로 관리하는 것은 짜증나는 일이니까 [`tf.merge_all_summaries`](../../api_docs/python/train.md#merge_all_summaries)를 사용해서 요약 노드들을 하나로 합쳐서 한 번에 모든 요약 데이터를 만들 수 있게 할 수 있습니다.

이제 통합된 요약 작업(summary op)을 실행시키면 모든 요약 데이터를 담은 `Summary` 프로토버퍼 오브젝트를 만들 수 있습니다. 마지막으로 이 요약 데이터를 디스크에 저장하기 위해 프로토버퍼 오브젝트를 [`tf.train.SummaryWriter`](../../api_docs/python/train.md#SummaryWriter)로 넘겨야 합니다.

`SummaryWriter`을 쓰기 위해서는 모든 요약 데이터를 저장할 디렉토리인 logdir을 정해줘야 합니다. `SummaryWriter`는 때에 따라 `그래프`도 이용할 수 있습니다. 만약 `그래프` 오브젝트를 이용하는 경우에는 TensorBoard가 텐서 형태(tensor shape) 정보에 덧붙여서 그래프도 보여줄 것입니다. 시각화된 그래프를 보면 그래프의 플로우에 대해서 더 잘 이해할 수 있겠죠. 자세한 내용은 [Tensor shape information](../../how_tos/graph_viz/index.md#tensor-shape-information)을 참조하세요.

드디어 그래프도 수정했고 `SummaryWriter`도 얻었습니다. 네트워크를 실행할 준비가 끝났어요! 원한다면 통합된 요약 작업을 매 단계마다 실행시켜서 엄청난 학습 데이터를 기록할 수 있습니다. 그런데 매 단계마다 저장하면 필요한 것보다 훨씬 많을 거에요. `n` 단계마다 요약 작업을 시키는 것을 고려해봅시다.

아래의 코드 예시는 [초보자를 위한 MNIST](http://tensorflow.org/tutorials/mnist/beginners/index.md)에 매 10단계마다 요약 작업을 하도록 조금 변형한 코드입니다. 이 코드를 실행시키고 `tensorboard --logdir=/tmp/mnist_logs`를 실행하면 학습하는 동안 가중치나 정확도같은 통계값이 어떻게 변했는지 볼 수 있습니다. 아래 코드는 일부를 발췌한 것이니 코드 전체는 [여기](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)를 참조하세요.

```python
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """간단한 신경망 레이어를 만들기 위한 재사용가능 코드
  
  행렬곱을 하고 편차(bias)를 더하고 비선형화를 위한 액티베이션 함수로 ReLU
  (역자 주: Rectified Linear Unit. 자주 유용하게 쓰이는 액티베이션 함수.)를 사용.
  가독성을 위해 name scoping을 했고 다량의 요약 작업을 추가.
  """
  # name scope를 추가하여 그래프의 계층들을 논리적으로 분류  
  with tf.name_scope(layer_name):
    # 레이어의 가중치를 저장할 변수    
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.scalar_summary('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

with tf.name_scope('cross_entropy'):
  diff = y_ * tf.log(y)
  with tf.name_scope('total'):
    cross_entropy = -tf.reduce_mean(diff)
  tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)

# 모든 요약 내용을 합치고 /tmp/mnist_logs에 기록합니다.(기본 경로)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
tf.initialize_all_variables().run()
```

`SummaryWriters`를 초기화 시킨 후에는 모델을 학습시키거나 테스트할 때 요약 내용(summary)을 `SummaryWriters`에 추가해야 합니다.

```python
# 모델을 학습시키고 요약 내용을 기록해 봅시다.
# 매 10 단계마다 테스트 세트의 정확도를 측정하고 그 요약 내용을 기록합니다.
# 매 단계마다 train_step을 실행하고 학습 내용을 추가합니다.

def feed_dict(train):
  """TensorFlow feed_dict를 만들기: data를 Tensor 플레이스홀더에 매칭"""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # 요약 내용과 테스트 세트 정확도를 기록합니다.
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # 학습한 세트에 대한 요약 내용을 기록하고 학습시킵니다.
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
```

이제 TensorBoard를 이용해서 이 데이터를 시각화할 준비가 끝났습니다.


## TensorBoard 실행

TensorBoard를 실행해보기 위해서 아래 명령을 이용해 봅시다.

```bash
tensorboard --logdir=path/to/log-directory
```

`logdir`은 데이터를 `SummaryWriter`가 데이터를 저장(serialize)해놓은 디렉토리를 가리킵니다. 만약 `logdir` 디렉토리에 다른 실행에 대한 데이터를 저장해놓은 하위 디렉토리가 있다면 TensorBoard는 모두 다 시각화해서 보여줄 것입니다. 한 번 TensorBoard가 실행되면 웹브라우저 주소창에 `localhost:6006`을 입력해서 볼 수 있습니다.

TensorBoard 화면 오른쪽 상단에서 네비게이션 탭을 찾을 수 있습니다. 각 탭들은 저장된 데이터 세트를 의미합니다.

그래프를 시각화하는 *그래프 탭*에 대한 더 자세한 정보는 [TensorBoard: 그래프 시각화](../../how_tos/graph_viz/index.md)를 참조하시기 바랍니다.

TensorBoard에 대한 전반적인 정보를 더 얻고 싶으시면 [TensorBoard 길라잡이](../../../tensorboard/README.md)를 참조해 주세요.
