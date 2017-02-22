# 분산환경 텐서플로우

이 문서는 텐서플로우 서버 클러스터를 생성하고, 클러스터 상에서 분산 처리를 수행하는 방법에 대하여 설명한다.  이 문서의 독자들은 텐서플로우를 이용한 [기본적인 프로그래밍 개념](../../get_started/basic_usage)은 숙지가 되어있다고 가정한다 .

## 안녕 분산환경 텐서플로우!

 텐서플로우 클러스터의 간단한 동작을 살펴보기 위해서는 아래 예제를 실행해보아라. 

```shell
# 싱글 프로세스 클러스터에서 텐서플로우 서버 실행하기.
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # 서버에서 세션 생성하기.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

[`tf.train.Server.create_local_server()`](../../api_docs/python/train.md#Server.create_local_server) 메소드는  in-process 서버 형태로 단일 프로세스 클러스터를 생성한다.

## 클러스터 생성하기

텐서플로우에서 "클러스터"란 텐서플로우 그래프 상에서의 분산 연산의 일부로서 "작업(Task)"의 집합을 의미한다. 각각의 작업은 텐서플로우의 "서버"에 연관되어 있으며, 각 서버는 세션을 생성할 수 있는 "마스터"와 그래프상에서 연산을 수행하는 "작업자"로 구성된다.   각 클러스터는 복 수개의 "직무(Jobs)"로 구성되어 있으며, 각각의 직무는 복 수개의 작업으로 이루어져 있다.

클러스터를 생성하기 위해서는, 작업 하나당 하나의 텐서플로우 서버를 실행해야한다. 각 작업은 보통 서로 다른 머신에서 실행되지만, 하나의 머신에서 여러개의 작업을 실행하는 것도 가능하다.(예를 들어 복 수개의 GPU를 사용하는 경우). 각 작업마다 아래의 절차를 따라서 진행된다.

1.  클러스터에 할당된 작업을 설명하는 **`tf.train.ClusterSpec`  을 생성**하라. 이 것은 각 작업마다 동일해야 한다.

2.  `tf.train.ClusterSpec` 를 생성자 인자로 넘겨주어 **`tf.train.Server` 를 생성**하고, 로컬 작업을 직무 이름과 작업 인덱스로 식별하라. 

      ​


### 클러스터를 설명하는 `tf.train.ClusterSpec` 생성하기

클러스터 명세 딕셔너리는 직무 이름과 네트워크 주소를 매핑한다. 아래와 같이 딕셔너리를 `tf.train.ClusterSpec` 의 생성자의 인자로 넘겨 주어라.

<table>
  <tr><th><code>tf.train.ClusterSpec</code> 생성자</th><th>사용 가능 작업</th>
  <tr>
    <td><pre>
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
</pre></td>
<td><code>/job:local/task:0<br/>/job:local/task:1</code></td>
  </tr>
  <tr>
    <td><pre>
tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222", 
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
</pre></td><td><code>/job:worker/task:0</code><br/><code>/job:worker/task:1</code><br/><code>/job:worker/task:2</code><br/><code>/job:ps/task:0</code><br/><code>/job:ps/task:1</code></td>
  </tr>
</table>

### 각 작업 마다 `tf.train.Server` 객체 생성하기

[`tf.train.Server`](../../api_docs/python/train.md#Server)  객체는 여러개의 로컬 디바이스 정보와, 각 작업과 디바이스를 연결해주는 정보인 `tf.train.ClusterSpec` , 분산 연산 수행에 이용되는 ["session target"](../../api_docs/python/client.md#Session) 을 포함하고 있다. 

각각의 서버는 특정한 이름이 부여된 직무의 멤버이며, 해당 직무에서의 작업 인덱스를 가지고 있다.  서버는 클러스터내에 있는 다른 서버와 통신이 가능하다.

예를 들어,  `localhost:2222` `localhost:2223` 두 개의 서버를 가진 클러스터를 구동하려면 밑의 두 코드를 로컬머신의 다른 두 개의 프로세스에서 실행하면 된다.

```python
# 0번 작업:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
```
```python
# 1번 작업:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```

**Note:**  거대한 클러스터를 생성하기 위해서 클러스터 명세를 일일히 수동으로 작성하는 것은 지루한 일일 수 있다. 우리는 이러한 작업을 프로그래밍으로 구동하기 위한 도구를 고민중이다. 예를 들어, [Kubernetes](http://kubernetes.io) 와 같은 클러스터 매니저를 사용하는 방법이다. 만약 우리가 지원하기를 원하는 클러스터 매니저가 있다면, [GitHub issue](https://github.com/tensorflow/tensorflow/issues) 에 제보해주길 바란다.

## 모델 내 에서 디바이스 명시하기

연산을 CPU 혹은 GPU에서 수행할지 선택할 때와 마찬가지로, 특정 연산에 연산을 수행할 디바이스를 명시할 때는     [`tf.device()`](../../api_docs/python/framework.md#device) 를 사용하면 된다. 

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```

위의 예제에서, 변수들은 `ps` 직무(job)에서 생성이 되고, 연산이 집중적으로 일어나는 모델은 `worker` 에서 생성이 된다. 텐서플로우는 각각의 직무간에 적절하게 데이터를 이동시켜준다.(정방향 연산시에는 `ps` 에서 `worker` 로 gradients 전파시에는 `worker` 에서 `ps`로 전달한다.)

## 훈련 복제

일반적으로 ''데이터 병렬화(data parallelism)'' 로 명명되는 훈련 방식은  `worker`직무의 여러개의 작업이 하나의 모델에 대하여, 데이터의 각기 다른 일부를 이용하여 `ps` 에서 생성된 공유변수를 병렬적으로 업데이트 시키는 방식이다. 모든 작업들은 각각 다른 머신에서 동작한다. 텐서플로우에서 이러한 훈련 방식을 구현하는 방법은 여러가지가 있는데, 우리는 복제된 모델을 간단하게 생성할 수 있도록 도와주는 라이브러리를 구축하였다. 시도 가능한 방법은 아래와 같다:

* **그래프내 복제(In-graph replication).** 이 방법에서 클라이언트는 한 세트의 변수( `/job:ps`에 연관된 `tf.Variable` )가 포함 된 `tf.Graph` 하나를 구축하고, `/job:worker` 에 소속된 서로 다른 작업에 각각 연관된 여러개의 연산 집중 모델을 복제하여 구축한다.
* **그래프간 복제(Between-graph replication).** 이 방법에서는 각 `/job:worker` 작업마다 별도의 클라이언트가 존재하며 일반적으로 연산수행 작업과 동일한 클라이언트에 있다. 각 클라이언트는 변수를 포함하는 유사한 그래프를 구축한다. (각 변수는 [`tf.train.replica_device_setter()`](https://github.com/JunYeopLee/tensorflow-kr/blob/master/g3doc/api_docs/python/train.md#replica_device_setter)를 사용하기 전에 `/job:ps`에 연관되어있고, 사용후에 동일한 작업에 매핑 된다.)  연산 집중 모델의 하나의 복사본은 `/job:worker` 의 로컬 작업에 연관되어 있다.
* **비동기식 훈련(Asynchronous training).** 이 방법에서는 각 그래프의 복제품이 독립적으로 각자 고유의 훈련 루프를 가지고 있다. 이 방법은 위의 두 복제방식과 호환이 가능하다.  
* **동기식 훈련(Synchronous training).** 이 방식에서는 각 그래프의 복제품이 현재의 변수에서 값을 읽어오고, 병렬적으로 gradient를 계산한뒤 병렬적으로 모델에 반영한다. 이 방법은 위 의 두 복제방식과 호환이 가능하다. 예를 들어 [CIFAR-10 multi-GPU trainer](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py) 에서 와 같이 gradient 평균을 활용하여 그래프내 복제를 하거나, `tf.train.SyncReplicasOptimizer`를 활용에서 그래프간 복제를 활용하는 방법이 있다.

### 총 정리 : 훈련 프로그램 예시

아래 코드는 그래프간 복제와 비동기식 훈련을 활용한 분산 훈련 프로그램의 뼈대 코드이다. 아래 코드에는 변수 서버(ps)와 연산 수행(worker)작업을 구현한 코드도 포함되어 있다. 

```python
import tensorflow as tf

# tf.train.ClusterSpec 정의를 위한 설정
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# tf.train.Server 정의를 위한 설정
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # 변수 서버와 작업자 클러스터를 생성한다.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
  # 로컬 작업 수행을 위해 서버를 생성하고 구동한다.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # 작업자에 연산을 분배한다.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # 모델 구축...
      loss = ...
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # 훈련 과정을 살펴보기 위해 "supervisor"를 생성한다.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # supervisor는 세션 초기화를 관리하고, checkpoint로부터 모델을 복원하고 
    # 에러가 발생하거나 연산이 완료되면 프로그램을 종료한다.
    with sv.managed_session(server.target) as sess:
      # "supervisor"가 종료되거나 1000000 step이 수행 될 때까지 반복한다.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # 훈련 과정을 비동기식으로 실행한다.Run a training step asynchronously.
        # 동기식 훈련 수행을 위해서는 `tf.train.SyncReplicasOptimizer`를 참조하라.
        _, step = sess.run([train_op, global_step])

    # 모든 서비스 중단.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
```

두 개의 변수 서버와 두 개의 연산 수행 작업으로 구성된 훈련용 프로그램을 구동하기 위해서는, 아래 커맨드 라인을 실행하면 된다.(스크립트 파일명이 `train.py`라고 가정한다.) :

```shell
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
```

## 용어 사전

<dl>
  <dt>클라이언트</dt>
  <dd>
    클라이언트는 일반적으로 텐서플로우 그래프를 정의하고, 클러스터와 상호작용하기 위해 `tensorflow::Session`을 구축하는 프로그램이다. 클라이언트는 보통 파이썬 또는 C++로 작성된다. 하나의 클라이언트 프로세스는 여러개의 텐서플로우 서버와 직접적으로 상호작용할 수 있으며(위의 "훈련 복제" 단원을 참조하라), 하나의 서버는 여러개의 클라이언트의 요청을 처리할 수 있다.
  </dd>
  <dt>클러스터</dt>
  <dd>
    텐서플로우 클러스터는 하나 혹은 복수개의 "직무(job)"로 구성 되며, 각 직무는 하나 혹은 복수개의 "작업(task)"으로 이루어 진다. 하나의 클러스터는 특정한 하나의 목표에 기여한다. 예를 들어 여러개의 머신을 활용하여 병렬적으로 신경망 회로를 훈련시키는 것을 들 수 있다.. 클러스터는 `tf.train.ClusterSpec` 객체를 이용하여 정의된다.
  </dd>
  <dt>직무(Job)</dt>
  <dd>
    하나의 직무(Job)은 여러 개의 작업(task)으로 이루어진다. 각 작업은 하나의 목적을 처리한다. 예를 들어 `ps(parameter server)`란 이름의 직무는 일반적으로 변수를 저장하는 주체이며, `worker`란 이름의 직무는 연산 집중 업무를 수행하는 노드를 담당한다. 직무에 포함되는 각 작업은 일반적으로 다른 머신에서 실행 된다. 각 직무가 담당하는 범위는 유동적으로 결정 된다. 예를 들어 `worker`가 변수를 저장하는 경우가 있을 수도 있다.
  </dd>
  <dt>마스터 서비스(Master service)</dt>
  <dd>
    분산된 디바이스에 원격 접근을 제공하고, 세션의 타겟으로 작동하는 RPC 서비스이다. 마스터 서비스는 <code>tensorflow::Session</code> 인터페이스를 구현하며, 복수개의 연산 수행 작업간의 조정을 담당하고 있다. 모든 텐서플로우 서버는 마스터 서비스를 구현해야 한다.
  </dd>
  <dt>작업(Task)</dt>
  <dd>
    하나의 작업은 하나의 텐서플로우 서버에 대응이 되고, 역시 하나의 프로세스와 대응된다. 하나의 작업은 특정한 직무에 소속되며, 직무 내의 작업 리스트에서는 작업 인덱스로 구분이 된다.
  </dd>
  <dt>텐서플로우 서버(TensorFlow server)</dt>
  <dd>
    <code>tf.train.Server</code> 를 수행하는 프로세스이다. 각 프로세스는 클러스터의 멤버이며 마스터 서비스와 작업 서비스를 제공한다.
  </dd>
  <dt>작업 서비스(Worker service)</dt>
  <dd>
    로컬 디바이스를 이용하여 텐서플로우 그래프의 한 부분 연산을 수행하는 RPC 서비스이다. 각 작업 서비스는 <a href=
    "https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto"
    ><code>worker_service.proto</code></a>를 구현한다. 모든 텐서플로우 서버는 작업서비스를 구현 한다.
  </dd>
</dl>
