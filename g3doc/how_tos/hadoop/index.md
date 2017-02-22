# 하둡에서 TensorFlow 돌리는 법(How to run TensorFlow on Hadoop)
(v0.12 rc1)

이 문서에서는 TensorFlow를 하둡에서 작동시키는 방법을 다룹니다. 앞으로 다양한 클러스터 관리 프로그램으로 확장되겠지만 여기서는 하둡 분산파일 시스템(HDFS)만 다루겠습니다.

## 하둡 분산파일 시스템(HDFS)

아래 내용은 [데이터 로딩](../reading_data/index.md)에 익숙한 분들을 위한 내용입니다.

HDFS를 TensorFlow와 함께 쓰기 위해 파일을 읽고 쓰는 경로(path)를 HDFS 경로로 바꿔줘야 합니다. 아래 예시를 보시죠.

```python
filename_queue = tf.train.string_input_producer([
    "hdfs://namenode:8020/path/to/file1.csv",
    "hdfs://namenode:8020/path/to/file2.csv",
])
```

자신의 HDFS 설정 파일들에 있는 네임노드(namenode, 역자 주: HDFS의 일종의 마스터 서버)를 쓰려면 파일의 접두사로 `hdfs://default/`를 붙여야 합니다.

TensorFlow 프로그램을 실행시킬 때 아래의 환경 변수가 설정되어 있어야 합니다:

*   **JAVA_HOME**: 자바가 설치된 위치
*   **HADOOP_HDFS_HOME**: HDFS가 설치된 위치. 아래 명령어를 실행시켜서 설정할 수도 있습니다:

```shell
source ${HADOOP_HOME}/libexec/hadoop-config.sh
```

*   **LD_LIBRARY_PATH**: 설치된 하둡 배포판(Hadoop distribution)이 `$HADOOP_HDFS_HOME/lib/native`에 libhdfs.so를 설치하지 않았으면 libjvm.so나 libhdfs.so의 경로를 포함하기 위해 필요합니다. 리눅스에서는 아래 명령어를 이용하세요:

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
```

*   **CLASSPATH**: TensorFlow 프로그램을 실행하기 전에 반드시 하둡 Jars가 추가되어 있어야 합니다. `${HADOOP_HOME}/libexec/hadoop-config.sh`로 설정된 CLASSPATH로는 충분하지 않으며 libhdfs 문서 대로 glob으로 확장되어야 합니다:

```shell
CLASSPATH=$($HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python your_script.py
```

[분산처리 TensorFlow](../distributed/index.md)를 이용하려면 모든 머신에 환경 변수 세트와 하둡이 있어야 합니다.