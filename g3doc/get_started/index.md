# 시작하기

텐서플로우를 실제로 작동시켜 봅시다!

시작하기 전에 앞으로 무엇을 배울지 힌트를 얻기위해 파이썬 API로 된 텐서플로우 코드를 잠깐 보겠습니다.

이 코드는 2차원 샘플 데이터를 사용하여 분포에 맞는 직선을 찾는(역주: 회귀분석) 간단한 파이썬 프로그램입니다.

```python
import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]

# Close the Session when we're done.
sess.close()
```

코드의 앞 부분은 데이터 플로우 그래프를 만들고 있습니다. 텐서플로우는 세션이 만들어져서 `run` 함수가 호출되기 전까지 어떤 것도 실제로 실행하지 않습니다.

좀 더 흥미를 돋우기 위해 전형적인 머신러닝 모델이 텐서플로우에서 어떻게 구현되는지 살펴보시면 좋습니다. 뉴럴 네트워크 분야에서 가장 전형적인 문제는 MNIST 손글씨 숫자를 분류하는 것입니다. 우리는 여기서 두가지 버전의 설명 즉 하나는 머신러닝 초보자를 위한 것과 하나는 전문가를 위한 버전을 제공합니다. 만약 다른 소프트웨어 패키지로 MNIST 모델을 여러번 훈련시킨 적이 있다면 붉은 알약을 선택하세요. 만약 MNIST에 대해 들어본 적이 없다면 푸른 알약을 선택하면 됩니다. 초보자와 전문가 사이의 어디라면 푸른색 말고 붉은 알약을 선택하시면 됩니다.

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px; display: flex; flex-direction: row">
 <a href="../tutorials/mnist/beginners/index.md" title="MNIST for ML Beginners tutorial">
   <img style="flex-grow:1; flex-shrink:1; border: 1px solid black;" src="../images/blue_pill.png" alt="MNIST for machine learning beginners tutorial" />
 </a>
 <a href="../tutorials/mnist/pros/index.md" title="Deep MNIST for ML Experts tutorial">
   <img style="flex-grow:1; flex-shrink:1; border: 1px solid black;" src="../images/red_pill.png" alt="Deep MNIST for machine learning experts tutorial" />
 </a>
</div>
<p style="font-size:10px;">Images licensed CC BY-SA 4.0; original by W. Carter</p>

바로 텐서플로우를 설치하고 배우고 싶다면 이 내용은 넘어가고 다음으로 진행해도 됩니다. 텐서플로우 기능을 설명하는 기술적인 튜토리얼에서 MNIST 예제를 또 사용하므로 다시 볼 수 있습니다.

## Recommended Next Steps
* [다운로드 및 설치](../get_started/os_setup.md)
* [기본적인 사용법](../get_started/basic_usage.md)
* [텐서플로우 구조](../tutorials/mnist/tf/index.md)
* [텐서플로우 플레이그라운드](http://playground.tensorflow.org)
