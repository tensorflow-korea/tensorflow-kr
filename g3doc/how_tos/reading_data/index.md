데이터 읽어들이기 (Reading data)
================================

TensorFlow 프로그램으로 데이터를 가져오는 방법에는 세 가지 주요 방법이 있다:

-	피딩(Feeding): 매 단계를 실행할 때마다 파이썬 코드가 데이터를 제공한다.
-	파일로부터 읽기(Reading from files): TensorFlow 그래프가 시작되는 부분에서 입력 파이프라인이 파일로부터 데이터를 읽어들인다.
-	미리 로드 된 데이터(Preloaded data): 작은 데이터셋의 경우 TensorFlow 그래프 내의 상수 또는 변수가 모든 데이터를 쥐고 있다.

[TOC]

피딩(Feeding)
-------------

TensorFlow의 피딩 기작은 계산 그래프 내의 모든 텐서에 데이터를 삽입할 수 있도록 한다. 따라서 파이썬으로도 그래프에 직접 데이터를 피딩할 수 있다.

계산을 개시하는 run()이나 eval()을 호출할 때, 피드 데이터를 제공하기 위해 `feed_dict` 전달인자를 사용하면 된다.

```python
with tf.Session():
  input = tf.placeholder(tf.float32)
  classifier = ...
  print(classifier.eval(feed_dict={input: my_python_preprocessing_fn()}))
```

변수와 상수를 포함하여 모든 텐서를 피드 데이터로 바꿀 수 있지만, 가장 좋은 방법은 [`placeholder` op](../../api_docs/python/io_ops.md#placeholder) 노드를 사용하는 것이다. `placeholder`는 오로지 피드를 받아들이기 위해 존재하는데, 초기화되지 않으며 데이터를 포함하지도 않는다. `placeholder`는 피드 없이 실행되는 경우 오류를 발생시켜 사용자가 피드하는 것을 잊지 않도록 한다.

`placeholder`를 사용하는 예제와 MNIST 데이터 학습을 위한 피딩은 [`tensorflow/examples/tutorials/mnist/fully_connected_feed.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/fully_connected_feed.py)에서 찾아볼 수 있으며, [MNIST tutorial](../../tutorials/mnist/tf/index.md) 에 설명이 있다.

파일로부터 읽기(Reading from files)
-----------------------------------

파일로부터 레코드를 읽어들이는 전형적인 파이프라인은 다음과 같은 단계를 갖는다:

1.	파일명 리스트 (The list of filenames)
2.	*선택적* 파일명 셔플링 (*Optional* filename shuffling)
3.	*선택적* 에폭 제한 (*Optional* epoch limit)
4.	파일명 큐 (Filename queue)
5.	파일 형식에 대한 리더기 (A reader for the file format)
6.	리더기로 읽어들인 레코드에 대한 해독기 (A decoder for a record read by the reader)
7.	*선택적* 전처리 (*Optional* preprocessing)
8.	예시 큐 (Example queue)

변수와 상수를 포함하여 모든 텐서를 피드 데이터로 바꿀 수 있지만, 가장 좋은 방법은 [`placeholder` op](../../api_docs/python/io_ops.md#placeholder) 노드를 사용하는 것이다.`placeholder`는 오로지 피드를 받아들이기 위해 존재하는데, 초기화되지 않으며 데이터를 포함하지도 않는다.`placeholder`는 피드 없이 실행되는 경우 오류를 발생시켜 사용자가 피드하는 것을 잊지 않도록 한다.

`placeholder`를 사용하는 예제와 MNIST 데이터 학습을 위한 피딩은 다음에서 찾아볼 수 있다: [`tensorflow/examples/tutorials/mnist/fully_connected_feed.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/fully_connected_feed.py), and is described in the [MNIST tutorial](../../tutorials/mnist/tf/index.md).

### 파일명, 셔플링, 에폭 제한

파일명 리스트의 경우 `["file0", "file1"]` 또는 `[("file%d" % i) for i in range(2)]`와 같이 문자열 상수 텐서를 사용하거나, [`tf.train.match_filenames_once` 함수](../../api_docs/python/io_ops.md#match_filenames_once)를 사용한다.

파일명 리스트를 [`tf.train.string_input_producer` 함수](../../api_docs/python/io_ops.md#string_input_producer)에 넘겨준다. `string_input_producer`는 리더기가 필요로 할 때까지 파일명을 유지하는 FIFO 큐를 만든다.

`string_input_producer`는 셔플링 및 최대 에폭 수 설정에 대한 옵션을 가지고 있다. 큐 실행기(queue runner)가 매 에폭마다 파일명 리스트를 큐에 한번 추가하면, `shuffle=True`로 설정된 경우 단일 에폭 내에서 파일명들이 뒤섞이게 된다. 이 절차는 파일이 균일하게 샘플링되도록 함으로써, 주어진 예시(example)가 다른 것에 비해 상대적으로 덜 추출되거나(under-sampling) 과하게 추출되지(over-sampling) 않도록 한다.

큐 실행기는, 파일명을 큐로부터 가져오는 리더기와 분리되어 있는 쓰레드에서 작동하기 때문에, 셔플링과 인큐잉(enqueuing, 큐에 집어넣기) 프로세스가 리더기를 블록(block)하지 않는다.

### 파일 형식(File formats)

입력 파일 형식에 적합한 리더기를 선택하고, 파일명 큐를 리더기의 read 메써드에 넘겨준다. read 메써드는 파일과 레코드를 식별하는 키(이상한 레코드를 얻은 경우 디버깅할 때 유용)와, 텐서가 아닌 스칼라 문자열 값을 출력한다. 이 문자열을, 어떤 예시를 구성하는 텐서로 해독하기 위해서는 하나 또는 여러 개의 해독기와 변환(conversion) 연산을 사용한다.

#### CSV 파일 (CSV files)

[CSV(comma-separated value, 쉼표로 구분된 값) 형식](https://tools.ietf.org/html/rfc4180)의 텍스트 파일을 읽기 위해서는, [`decode_csv`](../../api_docs/python/io_ops.md#decode_csv)연산과 함께 [`TextLineReader`](../../api_docs/python/io_ops.md#TextLineReader)를 사용한다. 예를 들어:

```python
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.pack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])

  coord.request_stop()
  coord.join(threads)
```

`read`는 파일에서 한 줄을 읽어들인다. 이후 `decode_csv` 연산은 읽어들인 결과를 텐서 리스트로 파싱한다. `record_defaults` 인수는 결과로 반환된 텐서들의 타입을 결정하고 입력 문자열에 값이 없는 경우 기본값을 설정한다.

`run` 또는 `eval`을 호출하여 `read`를 실행하기 전에 큐를 채우려면 반드시 `tf.train.start_queue_runners`를 호출해야 한다. 그렇지 않으면 `read`는 큐에서 파일 이름을 기다리는 상태로 블록된다.

#### Fixed length records

To read binary files in which each record is a fixed number of bytes, use[`tf.FixedLengthRecordReader`](../../api_docs/python/io_ops.md#FixedLengthRecordReader) with the [`tf.decode_raw`](../../api_docs/python/io_ops.md#decode_raw) operation. The `decode_raw` op converts from a string to a uint8 tensor.

For example, [the CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) uses a file format where each record is represented using a fixed number of bytes: 1 byte for the label followed by 3072 bytes of image data. Once you have a uint8 tensor, standard operations can slice out each piece and reformat as needed. For CIFAR-10, you can see how to do the reading and decoding in[`tensorflow/models/image/cifar10/cifar10_input.py`](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10_input.py) and described in[this tutorial](../../tutorials/deep_cnn/index.md#prepare-the-data).

#### Standard TensorFlow format

Another approach is to convert whatever data you have into a supported format. This approach makes it easier to mix and match data sets and network architectures. The recommended format for TensorFlow is a[TFRecords file](../../api_docs/python/python_io.md#tfrecords-format-details) containing[`tf.train.Example` protocol buffers](https://www.tensorflow.org/code/tensorflow/core/example/example.proto) (which contain[`Features`](https://www.tensorflow.org/code/tensorflow/core/example/feature.proto) as a field). You write a little program that gets your data, stuffs it in an`Example` protocol buffer, serializes the protocol buffer to a string, and then writes the string to a TFRecords file using the[`tf.python_io.TFRecordWriter` class](../../api_docs/python/python_io.md#TFRecordWriter). For example,[`tensorflow/examples/how_tos/reading_data/convert_to_records.py`](https://www.tensorflow.org/code/tensorflow/examples/how_tos/reading_data/convert_to_records.py) converts MNIST data to this format.

To read a file of TFRecords, use[`tf.TFRecordReader`](../../api_docs/python/io_ops.md#TFRecordReader) with the [`tf.parse_single_example`](../../api_docs/python/io_ops.md#parse_single_example) decoder. The `parse_single_example` op decodes the example protocol buffers into tensors. An MNIST example using the data produced by `convert_to_records` can be found in[`tensorflow/examples/how_tos/reading_data/fully_connected_reader.py`](https://www.tensorflow.org/code/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py), which you can compare with the `fully_connected_feed` version.

### Preprocessing

You can then do any preprocessing of these examples you want. This would be any processing that doesn't depend on trainable parameters. Examples include normalization of your data, picking a random slice, adding noise or distortions, etc. See[`tensorflow/models/image/cifar10/cifar10.py`](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10.py) for an example.

### Batching

At the end of the pipeline we use another queue to batch together examples for training, evaluation, or inference. For this we use a queue that randomizes the order of examples, using the[`tf.train.shuffle_batch` function](../../api_docs/python/io_ops.md#shuffle_batch).

Example:

```
def read_my_file_format(filename_queue):
  reader = tf.SomeReader()
  key, record_string = reader.read(filename_queue)
  example, label = tf.some_decoder(record_string)
  processed_example = some_processing(example)
  return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch
```

If you need more parallelism or shuffling of examples between files, use multiple reader instances using the[`tf.train.shuffle_batch_join` function](../../api_docs/python/io_ops.md#shuffle_batch_join). For example:

```
def read_my_file_format(filename_queue):
  # Same as above

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example_list = [read_my_file_format(filename_queue)
                  for _ in range(read_threads)]
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch
```

You still only use a single filename queue that is shared by all the readers. That way we ensure that the different readers use different files from the same epoch until all the files from the epoch have been started. (It is also usually sufficient to have a single thread filling the filename queue.)

An alternative is to use a single reader via the[`tf.train.shuffle_batch` function](../../api_docs/python/io_ops.md#shuffle_batch) with `num_threads` bigger than 1. This will make it read from a single file at the same time (but faster than with 1 thread), instead of N files at once. This can be important:

-	If you have more reading threads than input files, to avoid the risk that you will have two threads reading the same example from the same file near each other.
-	Or if reading N files in parallel causes too many disk seeks.

How many threads do you need? the `tf.train.shuffle_batch*` functions add a summary to the graph that indicates how full the example queue is. If you have enough reading threads, that summary will stay above zero. You can[view your summaries as training progresses using TensorBoard](../../how_tos/summaries_and_tensorboard/index.md).

### Creating threads to prefetch using `QueueRunner` objects

The short version: many of the `tf.train` functions listed above add[`QueueRunner`](../../api_docs/python/train.md#QueueRunner) objects to your graph. These require that you call[`tf.train.start_queue_runners`](../../api_docs/python/train.md#start_queue_runners) before running any training or inference steps, or it will hang forever. This will start threads that run the input pipeline, filling the example queue so that the dequeue to get the examples will succeed. This is best combined with a[`tf.train.Coordinator`](../../api_docs/python/train.md#Coordinator) to cleanly shut down these threads when there are errors. If you set a limit on the number of epochs, that will use an epoch counter that will need to be initialized. The recommended code pattern combining these is:

```python
# Create the graph, etc.
init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run(train_op)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
```

#### Aside: What is happening here?

First we create the graph. It will have a few pipeline stages that are connected by queues. The first stage will generate filenames to read and enqueue them in the filename queue. The second stage consumes filenames (using a`Reader`), produces examples, and enqueues them in an example queue. Depending on how you have set things up, you may actually have a few independent copies of the second stage, so that you can read from multiple files in parallel. At the end of these stages is an enqueue operation, which enqueues into a queue that the next stage dequeues from. We want to start threads running these enqueuing operations, so that our training loop can dequeue examples from the example queue.

<div style="width:70%; margin-left:12%; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/AnimatedFileQueues.gif">
</div>

The helpers in `tf.train` that create these queues and enqueuing operations add a [`tf.train.QueueRunner`](../../api_docs/python/train.md#QueueRunner) to the graph using the[`tf.train.add_queue_runner`](../../api_docs/python/train.md#add_queue_runner) function. Each `QueueRunner` is responsible for one stage, and holds the list of enqueue operations that need to be run in threads. Once the graph is constructed, the[`tf.train.start_queue_runners`](../../api_docs/python/train.md#start_queue_runners) function asks each QueueRunner in the graph to start its threads running the enqueuing operations.

If all goes well, you can now run your training steps and the queues will be filled by the background threads. If you have set an epoch limit, at some point an attempt to dequeue examples will get an[`tf.OutOfRangeError`](../../api_docs/python/client.md#OutOfRangeError). This is the TensorFlow equivalent of "end of file" (EOF) -- this means the epoch limit has been reached and no more examples are available.

The last ingredient is the[`Coordinator`](../../api_docs/python/train.md#Coordinator). This is responsible for letting all the threads know if anything has signalled a shut down. Most commonly this would be because an exception was raised, for example one of the threads got an error when running some operation (or an ordinary Python exception).

For more about threading, queues, QueueRunners, and Coordinators[see here](../../how_tos/threading_and_queues/index.md).

#### Aside: How clean shut-down when limiting epochs works

Imagine you have a model that has set a limit on the number of epochs to train on. That means that the thread generating filenames will only run that many times before generating an `OutOfRange` error. The QueueRunner will catch that error, close the filename queue, and exit the thread. Closing the queue does two things:

-	Any future attempt to enqueue in the filename queue will generate an error. At this point there shouldn't be any threads trying to do that, but this is helpful when queues are closed due to other errors.
-	Any current or future dequeue will either succeed (if there are enough elements left) or fail (with an `OutOfRange` error) immediately. They won't block waiting for more elements to be enqueued, since by the previous point that can't happen.

The point is that when the filename queue is closed, there will likely still be many filenames in that queue, so the next stage of the pipeline (with the reader and other preprocessing) may continue running for some time. Once the filename queue is exhausted, though, the next attempt to dequeue a filename (e.g. from a reader that has finished with the file it was working on) will trigger an`OutOfRange` error. In this case, though, you might have multiple threads associated with a single QueueRunner. If this isn't the last thread in the QueueRunner, the `OutOfRange` error just causes the one thread to exit. This allows the other threads, which are still finishing up their last file, to proceed until they finish as well. (Assuming you are using a[`tf.train.Coordinator`](../../api_docs/python/train.md#Coordinator), other types of errors will cause all the threads to stop.) Once all the reader threads hit the `OutOfRange` error, only then does the next queue, the example queue, gets closed.

Again, the example queue will have some elements queued, so training will continue until those are exhausted. If the example queue is a[`RandomShuffleQueue`](../../api_docs/python/io_ops.md#RandomShuffleQueue), say because you are using `shuffle_batch` or `shuffle_batch_join`, it normally will avoid ever going having fewer than its `min_after_dequeue` attr elements buffered. However, once the queue is closed that restriction will be lifted and the queue will eventually empty. At that point the actual training threads, when they try and dequeue from example queue, will start getting `OutOfRange` errors and exiting. Once all the training threads are done,[`tf.train.Coordinator.join`](../../api_docs/python/train.md#Coordinator.join) will return and you can exit cleanly.

### Filtering records or producing multiple examples per record

Instead of examples with shapes `[x, y, z]`, you will produce a batch of examples with shape `[batch, x, y, z]`. The batch size can be 0 if you want to filter this record out (maybe it is in a hold-out set?), or bigger than 1 if you are producing multiple examples per record. Then simply set `enqueue_many=True` when calling one of the batching functions (such as `shuffle_batch` or`shuffle_batch_join`).

### Sparse input data

SparseTensors don't play well with queues. If you use SparseTensors you have to decode the string records using[`tf.parse_example`](../../api_docs/python/io_ops.md#parse_example) **after** batching (instead of using `tf.parse_single_example` before batching).

Preloaded data
--------------

This is only used for small data sets that can be loaded entirely in memory. There are two approaches:

-	Store the data in a constant.
-	Store the data in a variable, that you initialize and then never change.

Using a constant is a bit simpler, but uses more memory (since the constant is stored inline in the graph data structure, which may be duplicated a few times).

```python
training_data = ...
training_labels = ...
with tf.Session():
  input_data = tf.constant(training_data)
  input_labels = tf.constant(training_labels)
  ...
```

To instead use a variable, you need to also initialize it after the graph has been built.

```python
training_data = ...
training_labels = ...
with tf.Session() as sess:
  data_initializer = tf.placeholder(dtype=training_data.dtype,
                                    shape=training_data.shape)
  label_initializer = tf.placeholder(dtype=training_labels.dtype,
                                     shape=training_labels.shape)
  input_data = tf.Variable(data_initializer, trainable=False, collections=[])
  input_labels = tf.Variable(label_initializer, trainable=False, collections=[])
  ...
  sess.run(input_data.initializer,
           feed_dict={data_initializer: training_data})
  sess.run(input_labels.initializer,
           feed_dict={label_initializer: training_labels})
```

Setting `trainable=False` keeps the variable out of the`GraphKeys.TRAINABLE_VARIABLES` collection in the graph, so we won't try and update it when training. Setting `collections=[]` keeps the variable out of the`GraphKeys.VARIABLES` collection used for saving and restoring checkpoints.

Either way,[`tf.train.slice_input_producer function`](../../api_docs/python/io_ops.md#slice_input_producer) can be used to produce a slice at a time. This shuffles the examples across an entire epoch, so further shuffling when batching is undesirable. So instead of using the `shuffle_batch` functions, we use the plain[`tf.train.batch` function](../../api_docs/python/io_ops.md#batch). To use multiple preprocessing threads, set the `num_threads` parameter to a number bigger than 1.

An MNIST example that preloads the data using constants can be found in[`tensorflow/examples/how_tos/reading_data/fully_connected_preloaded.py`](https://www.tensorflow.org/code/tensorflow/examples/how_tos/reading_data/fully_connected_preloaded.py), and one that preloads the data using variables can be found in[`tensorflow/examples/how_tos/reading_data/fully_connected_preloaded_var.py`](https://www.tensorflow.org/code/tensorflow/examples/how_tos/reading_data/fully_connected_preloaded_var.py), You can compare these with the `fully_connected_feed` and`fully_connected_reader` versions above.

Multiple input pipelines
------------------------

Commonly you will want to train on one dataset and evaluate (or "eval") on another. One way to do this is to actually have two separate processes:

-	The training process reads training input data and periodically writes checkpoint files with all the trained variables.
-	The evaluation process restores the checkpoint files into an inference model that reads validation input data.

This is what is done in[the example CIFAR-10 model](../../tutorials/deep_cnn/index.md#save-and-restore-checkpoints). This has a couple of benefits:

-	The eval is performed on a single snapshot of the trained variables.
-	You can perform the eval even after training has completed and exited.

You can have the train and eval in the same graph in the same process, and share their trained variables. See[the shared variables tutorial](../../how_tos/variable_scope/index.md).
