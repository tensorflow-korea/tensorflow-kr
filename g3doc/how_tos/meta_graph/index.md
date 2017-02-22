# 메타 그래프를 내보내고 가져오기

[`MetaGraph`](https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto)는 텐서플로우 GraphDef 뿐만 아니라
프로세스 경계를 교차할 때 그래프에서 연산을 실행하는 데 필요한 관련된 metadata도 포함합니다.
그것은 또한 그래프의 장기간 보관에도 사용될 수 있습니다.
MetaGraph는 훈련을 계속하는 데 필요한 정보를 포함하고 있으며, 이전에 훈련된 그래프에 대한 평가를 수행하거나 실행하는 데 필요한 정보를 수록하고 있습니다.

전체 모델을 내보내고 가져오는 API는 
[`tf.train.Saver`](../../api_docs/python/state_ops.md#Saver) class에
[`export_meta_graph`](../../api_docs/python/train.md#export_meta_graph)와
[`import_meta_graph`](../../api_docs/python/train.md#import_meta_graph)입니다.


## MetaGraph에 있는 것

MetaGraph에 수록된 정보는
[`MetaGraphDef`](https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto)
프로토콜 버퍼를 나타냅니다. 다음 필드가 포함됩니다.

* 버전과 기타 사용자 정보 같은 메타 정보에 대한 [`MetaInfoDef`](https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto).
* 그래프를 묘사하기 위한 [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).
* 세이버(saver)에 대한 [`SaverDef`](https://www.tensorflow.org/code/tensorflow/core/protobuf/saver.proto).
* [`CollectionDef`](https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto)
map은 모델의 [`Variables`](https://tensorflow.org/api_docs/python/state_ops.html),
[`QueueRunners`](https://tensorflow.org/api_docs/python/train.html#QueueRunner), etc와 같은 추가적인 요소를 더 자세히 설명합니다.
Python 오브젝트를 `MetaGraphDef`로부터 직렬화하기 위해서, Python 클래스는 `to_proto()`와 `from_proto()`메소드를 실행하고,
`register_proto_function`를 사용해서 시스템에 등록합니다.

  예를 들어,

  ```Python
  def to_proto(self):
    """Converts a `Variable` to a `VariableDef` protocol buffer.

    Returns:
      A `VariableDef` protocol buffer.
    """
    var_def = variable_pb2.VariableDef()
    var_def.variable_name = self._variable.name
    var_def.initializer_name = self.initializer.name
    var_def.snapshot_name = self._snapshot.name
    if self._save_slice_info:
      var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto())
    return var_def

  @staticmethod
  def from_proto(variable_def):
    """Returns a `Variable` object created from `variable_def`."""
    return Variable(variable_def=variable_def)

  ops.register_proto_function(ops.GraphKeys.VARIABLES,
                              proto_type=variable_pb2.VariableDef,
                              to_proto=Variable.to_proto,
                              from_proto=Variable.from_proto)
  ```

## 전체 모델을 MetaGraph로 내보내기

실행 중인 모델을 MetaGraph로 내보내는 API는 `export_meta_graph()`입니다.

  ```Python
  def export_meta_graph(filename=None, collection_list=None, as_text=False):
    """Writes `MetaGraphDef` to save_path/filename.

    Args:
      filename: Optional meta_graph filename including the path.
      collection_list: List of string keys to collect.
      as_text: If `True`, writes the meta_graph as an ASCII proto.

    Returns:
      A `MetaGraphDef` proto.
    """
  ```

  `collection`은 사용자가 고유하게 식별하고 쉽게 검색할 수 있는 Python 객체를 포함할 수 있습니다.
  이 객체들은 그래프 안에서 `train_op`나 하이퍼 파라미터(hyper parameters), "learning rate"처럼 특별한 연산을 할 수 있습니다.
  사용자는 내보내려는 컬렉션 목록을 지정할 수 있습니다. 만약 `collection_list`가 지정되지 않으면,
  모델 안에 모든 컬렉션이 내 보내어질 겁니다.

  API는 직렬화된 프로토콜 버퍼를 반환합니다. 만약 `filename`이 지정됐다면, 프로토콜 버퍼는 파일에 쓰여질 겁니다.

  다음은 일반적인 사용 모델의 일부입니다.

  * 기본 실행 그래프 내보내기:

  ```Python
  # Build the model
  ...
  with tf.Session() as sess:
    # Use the model
    ...
  # Export the model to /tmp/my-model.meta.
  meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')
  ```

  * 기본 실행 그래프와 컬렉션의 일부분만 내보냅니다.

  ```Python
  meta_graph_def = tf.train.export_meta_graph(
      filename='/tmp/my-model.meta',
      collection_list=["input_tensor", "output_tensor"])
  ```

MetaGraph는 또한 
[`tf.train.Saver`](../../api_docs/python/state_ops.md#Saver)에 `save()` API를 통해 자동으로 내보내기 됩니다.


## MetaGraph 가져오기

MetaGraph 파일을 그래프로 가져오기위한 API는 `import_meta_graph()`입니다.

다음은 일반적인 사용 모델의 일부입니다:

* 처음부터 모델을 구축하지 않고 가져와서 계속 훈련합니다.

  ```Python
  ...
  # Create a saver.
  saver = tf.train.Saver(...variables...)
  # Remember the training_op we want to run by adding it to a collection.
  tf.add_to_collection('train_op', train_op)
  sess = tf.Session()
  for step in xrange(1000000):
      sess.run(train_op)
      if step % 1000 == 0:
          # Saves checkpoint, which by default also exports a meta_graph
          # named 'my-model-global_step.meta'.
          saver.save(sess, 'my-model', global_step=step)
  ```

  나중에 우리는 처음부터 모델을 구축하지 않고 저장된 meta_graph로부터 계속 훈련할 수 있습니다.

  ```Python
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() returns a list. In this example we only want the
    # first one.
    train_op = tf.get_collection('train_op')[0]
    for step in xrange(1000000):
      sess.run(train_op)
  ```

* 그래프를 가져오고 확장하십시오.

  예를 들어, 먼저 추론 그래프를 작성하여 메타 그래프로 내보낼 수 있습니다.

  ```Python
  # Creates an inference graph.
  # Hidden 1
  images = tf.constant(1.2, tf.float32, shape=[100, 28])
  with tf.name_scope("hidden1"):
    weights = tf.Variable(
        tf.truncated_normal([28, 128],
                            stddev=1.0 / math.sqrt(float(28))),
        name="weights")
    biases = tf.Variable(tf.zeros([128]),
                         name="biases")
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope("hidden2"):
    weights = tf.Variable(
        tf.truncated_normal([128, 32],
                            stddev=1.0 / math.sqrt(float(128))),
        name="weights")
    biases = tf.Variable(tf.zeros([32]),
                         name="biases")
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope("softmax_linear"):
    weights = tf.Variable(
        tf.truncated_normal([32, 10],
                            stddev=1.0 / math.sqrt(float(32))),
        name="weights")
    biases = tf.Variable(tf.zeros([10]),
                         name="biases")
    logits = tf.matmul(hidden2, weights) + biases
    tf.add_to_collection("logits", logits)

  init_all_op = tf.initialize_all_variables()

  with tf.Session() as sess:
    # Initializes all the variables.
    sess.run(init_all_op)
    # Runs to logit.
    sess.run(logits)
    # Creates a saver.
    saver0 = tf.train.Saver()
    saver0.save(sess, saver0_ckpt)
    # Generates MetaGraphDef.
    saver0.export_meta_graph('my-save-dir/my-model-10000.meta')
  ```

  그런 다음 나중에 가져와서 훈련 그래프로 확장합니다.

  ```Python
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # Addes loss and train.
    labels = tf.constant(0, tf.int32, shape=[100], name="labels")
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 10]), 1.0, 0.0)
    logits = tf.get_collection("logits")[0]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

    tf.scalar_summary(loss.op.name, loss)
    # Creates the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    # Runs train_op.
    train_op = optimizer.minimize(loss)
    sess.run(train_op)
  ```

* 하이퍼 파라미터(Hyper Parameters) 검색

  ```Python
  filename = ".".join([tf.latest_checkpoint(train_dir), "meta"])
  tf.train.import_meta_graph(filename)
  hparams = tf.get_collection("hparams")
  ```
