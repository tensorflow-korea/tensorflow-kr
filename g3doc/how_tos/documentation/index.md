# TensorFlow 문서 작성

TensorFlow의 문서는 
[Markdown](https://daringfireball.net/projects/markdown/)에서 유지되고 있고 
`g3doc/` 디렉토리에 존재합니다. *Introduction*, *Overview*, *Tutorials*, *How-Tos*
부분은 수작업으로 수정됩니다.

`g3doc/api_docs` 디렉토리에 있는 어떤 것이든 코드에 있는 주석으로부터 생성되기 때문에
직접 수정해선 안됩니다. `tools/docs/gen_docs.sh` 스크립트는
API문서를 생성합니다. 스크립트를 인자 없이 호출하면 Python API 문서만을 재작성합니다. 
(Ops에 관한 문서는 Python, C++ 상관없이 작성됩니다.). `-a` 를 인자로 전달하면 
C++ 문서 또한 재작성합니다. 이것은 반드시 `tools/docs` 디렉토리에서 호출되어야 합니다. 
그리고 `-a` 를 인자로 전달하려면 `doxygen` 의 설치가 요구됩니다.

## Python API 문서

Ops, classes, utility 함수는 `image_ops.py` 와 같은 Python 모듈에 정의되어 있습니다. 
그 모듈의 docstring은 해당 Python 파일에 대해 생성되는 마크다운 파일의 시작부분에 삽입됩니다. 
그래서 `image_ops.md` 는 `image_ops.py` 모듈에 있는 docstring으로 시작합니다. 
`python/framework/gen_docs_combined.py` 는 마크다운 파일이 생성되는 모든 _libraries_(_라이브러리_) 목록을 포함하고 있습니다. 
만약에 새로운 라이브러리를 더한다면 (API 문서에 독립된 섹션을 생성하는 것), 
반드시 `gen_docs_combined.py` 에 있는 라이브러리 목록에 추가해야 합니다. 
C++ API는 하나의 라이브러리 파일만 존재합니다, 그것의 마크다운은 `api_docs/cc/index.md` 에서 생성된 `gen_cc_md.py` 의 문자열입니다. 
나머지 C++문서들은 doxygen에 의해 만들어진 XML파일로부터 생성됩니다.

라이브러리로 등록된 파일의 모듈 docstring에서, (빈 줄의 시작 부분에) `@@<python-name>` 문법으로 
Ops, classes, functions를 호출해 그것들을 위해 생성된 문서를 삽입할 수 있습니다. 
호출한 op, function 또는 class는 같은 파일에서 정의될 필요가 없습니다.
이것은 Ops, classes, functions가 기록되는 순서를 정할 수 있게 해줍니다.
고수준의 문서를 적절히 배치해서 논리적인 순서로 나누세요. 

모든 공개된 op, class, function 은 반드시 라이브러리의 서두에서 `@@` 로 호출되어야 합니다. 
그렇게 하지 않을 경우 `doc_gen_test` 가 실패하게 됩니다.

Ops를 위한 문서는 자동적으로 Python wrapper 또는 C++ Ops registrations로 부터 발췌합니다. 
둘 중 Python wrappers를 우선적으로 가져옵니다.

* Python wrappers는 `python/ops/*.py` 에 있습니다.
* C++ Ops registrations는 `core/ops/*.cc` 에 있습니다.

Classes와 Utility Functions를 위한 문서는 docstring에서 발췌합니다.

## Op 문서 스타일 가이드

이상적으로는, 제시된 순서를 따라 아래와 같은 정보를 제공해야 합니다:

* op가 무엇을 하는지를 설명하는 짧은 문장.
* op에 인자를 전달할 때 어떤 일이 생기는지에 대한 짧은 설명.
* op가 어떻게 작동하는지를 보여주는 예시(수도코드가 가장 좋음).
* 요구사항, 경고, 중요한 내용 (하나라도 있다면).
* 입력, 출력, 속성, op 생성자의 다른 변수에 대한 설명.

위 항목에 대해 설명된 자세한 정보는
[이곳](#description-of-the-docstring-sections).

글을 마크다운 (.md) 포멧으로 적으세요. 기본적인 문법에 관한 레퍼런스는
[여기](https://daringfireball.net/projects/markdown/)에 있습니다. 방정식 표기를 위해 
[MathJax](https://www.mathjax.org)을 사용할 수 있습니다. 이것들은 
[tensorflow.org](https://www.tensorflow.org)에서 적절히 표현됩니다. 하지만
[github](https://github.com/tensorflow/tensorflow)에 나타나지는 않습니다.

### 코드에 대해서 적을 때

문서에서 아래와 같은 것들을 적을 때 ` ` (backticks) 로 감싸야 합니다:

- 인자의 이름 (e.g. `input`, `x`, `tensor`)
- 반환되는 Tensor의 이름 (e.g. `output`, `idx`, `out`)
- 데이터 타입 (e.g. `int32`, `float`, `uint8`)
- 문서에서 언급되는 다른 op의 이름 (e.g. `list_diff()`, `shuffle()`)
- 클래스 이름 (e.g. 실제로 `Tensor` 객체를 의미할 때만 `Tensor` 를 사용하세요.
  op가 tensor, 그래프, 실행에서 보통 어떻게 작동하는지 설명할 때는 대문자나 backticks를 사용하면 안됩니다.)
- 파일 이름 (e.g. `image_ops.py`, `/path-to-your-data/xml/example-name`)

예시 코드와 수도코드는 backticks를 세번 써서 감싸야 합니다.
op가 무엇을 반환하는지 보여주고 싶을 때는 single equal sign( '=' 기호 )대신 `==>`를 사용해야 합니다.
예시:

    ```
    # 'input' is a tensor of shape [2, 3, 5]
    (tf.expand_dims(input, 0)) ==> [1, 2, 3, 5]
    ```

Python 코드 예시를 제공하고자 한다면 문법 강조가 적절히 되도록 Python 스타일 라벨을 추가하세요:

```markdown
 ```python
 # some Python code
 ```
```

수식이나 조건은 하나의 backticks로 감싸세요. 예시:

```markdown
This operation requires that `-1-input.dims() <= dim <= input.dims()`.
```

### Tensor의 차원

보통 tensor에 대해 말할 때는 tensor라는 단어의 첫 글자를 대문자로 쓰지 마세요.
op에 인자로 전달하거나 op에서 반환하는 특정한 객체를 말할 때는 Tensor의 첫 글자를 대문자로 쓰고 
backticks로 감싸세요. 왜냐 하면 전달되어지는 `Tensor` 객체에 대해 말하고 있기 때문입니다.

진짜 `Tensors` 라는 객체에 대해 이야기하지 않는다면 `Tensors`를 여러개의 Tensor 객체를 서술할 때 사용하지 마세요.
"a list of `Tensor`objects." (`Tensor` 객체 리스트) 또는  "`Tensor`s" (`Tensor`들)로 부르는게 낫습니다.

tensor의 크기에 대해 이야기 할 때는 아래의 가이드라인을 참고하세요:

tensor의 크기를 말할 때는 "dimension"(차원) 이라는 단어를 사용하세요. 
크기를 특정화할 필요가 있다면 아래의 규칙을 사용하세요:

- scalar(스칼라)는 "0-D tensor" 로 부른다
- vector(벡터)는 "1-D tensor" 로 부른다
- matrix(메트릭스)는 "2-D tensor" 로 부른다
- 3차원 이상인 tensors는 "3-D tensors 또는 n-D tensors" 로 부른다. 
  이해가 될 경우에는 "rank" 라는 단어를 사용한다. 하지만 "dimension"을 대신 쓰도록 노력하라. 
  절대 tensor의 크기를 표현하기 위해 "order" 라는 단어를 사용하지 말라.

tensor의 dimensions를 자세히 설명 하려면 "shape" 라는 단어를 사용하세요. 
그리고 backticks 와 꺽쇠 괄호를 사용해서 모양을 보여주세요.
예시:

```markdown
If `input` is a 3-D tensor with shape `[3, 4, 3]`, this operation will return
a 3-D tensor with shape `[6, 8, 6]`.
```

### 링크

`g3docs` 트리에 있는 다른 것들에 링크를 걸려면 
`[tf.parse_example](../api_docs/python/ops.md#parse_example)` 처럼 상대 경로를 사용하세요.
내부 링크에 절대 경로를 사용하지 마세요. 왜냐하면 웹사이트 생성기를 손상시키기 때문입니다.

소스코드에 링크를 걸려면 `https://www.tensorflow.org/code/` 로 시작하고 깃허브 루트에서 시작하는 
파일 이름으로 이어지는 링크를 사용하세요. 
예를 들면, 이 파일로 연결된 링크는 `https://www.tensorflow.org/code/tensorflow/g3doc/how_tos/documentation/index.md` 
처럼 쓰여져야 합니다.
이것은 [tensorflow.org](https://www.tensorflow.org/) 
가 당신이 보고 있는 문서의 버전에 대응한 코드의 브랜치를 가리킬 수 있도록 보장해 줍니다.
url 파라미터를 URL에 포함하지 마세요.


### C++에서 정의된 Ops

C++에 정의된 모든 Ops는 반드시 `REGISTER_OP` 선언 부분에 기록되어 있어야 합니다. 
C++ 파일에 있는 docstring은 입력 타입, 출력 타입, Attr 타입, default 값 정보를 자동적으로 
추가하기 위해 처리됩니다.

예시:

```c++
REGISTER_OP("PngDecode")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .Doc(R"doc(
Decodes the contents of a PNG file into a uint8 tensor.

contents: PNG file contents.
channels: Number of color channels, or 0 to autodetect based on the input.
  Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.
  If the input has a different number of channels, it will be transformed
  accordingly.
image:= A 3-D uint8 tensor of shape `[height, width, channels]`.
  If `channels` is 0, the last dimension is determined
  from the png contents.
)doc");
```

이 부분의 마크다운의 결과:

```markdown
### tf.image.png_decode(contents, channels=None, name=None) {#png_decode}

Decodes the contents of a PNG file into a uint8 tensor.

#### Args:

*  <b>contents</b>: A string Tensor. PNG file contents.
*  <b>channels</b>: An optional int. Defaults to 0.
    Number of color channels, or 0 to autodetect based on the input.
    Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.  If the
    input has a different number of channels, it will be transformed accordingly.
*  <b>name</b>: A name for the operation (optional).

#### Returns:

  A 3-D uint8 tensor of shape `[height, width, channels]`.
  If `channels` is 0, the last dimension is determined
  from the png contents.
```

인자에 대한 설명은 대부분 자동적으로 추가됩니다. 특히 doc generator(문서 생성기)는 
자동적으로 모든 입력, attrs, 출력의 이름과 타입을 추가합니다.
위의 예시에서 `<b>contents</b>: A string Tensor.` 는 자동으로 추가되었습니다.
글이 자연스럽게 흘러가도록 추가적인 문장을 해당 설명 뒤에 써야 합니다. 

입,출력에 대해서, equal sign( = 기호)을 추가적인 문장의 서두에 붙여서 이름과 타입을 자동으로 추가하는 것을 
막을 수 있습니다. 위의 예시에서, 이름이 `image` 인 출력에 대한 설명은 우리가 입력한 글 `A 3-D uint8 Tensor...` 이전에 
`A uint8 Tensor.` 가 추가되는 것을 방지하기 위해 `=` 로 시작합니다.
이 방법으로는 attrs의 이름, 타입, default 값이 추가되는 것을 막을 수는 없기 때문에 글을 적을 때 신중해야 합니다.

### Python에서 정의된 Ops

op가 `python/ops/*.py` 파일에 정의되어 있다면, 모든 인자와 출력값(반환값) tensor에 관한 글을 제공해야 합니다.

Python docstring 규칙에 따라야 하고 docstring에서 마크다운을 사용해야 합니다. 
doc generator(문서 생성기)는 Python에 정의된 ops에 관한 글은 어떤 것도 자동으로 생성하지 않기 때문에 
당신이 적은 것을 사용합니다.

간단한 예시:

```python
def foo(x, y, name="bar"):
  """Computes foo.

  Given two 1-D tensors `x` and `y`, this operation computes the foo.

  For example:

  ```
  # x is [1, 1]
  # y is [2, 2]
  tf.foo(x, y) ==> [3, 3]
  ```

  Args:
    x: A `Tensor` of type `int32`.
    y: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32` that is the foo of `x` and `y`.

  Raises:
    ValueError: If `x` or `y` are not of type `int32`.
  """

  ...
```

## Docstring 부분에 관한 설명

여기에 더 자세한 내용과 docstring의 각 속성에 대한 예시가 있습니다.

### op가 무엇을 하는지 설명하는 짧은 문장

예시:

```markdown
Concatenates tensors.
```

```markdown
Flips an image horizontally from left to right.
```

```markdown
Computes the Levenshtein distance between two sequences.
```

```markdown
Saves a list of tensors to a file.
```

```markdown
Extracts a slice from a tensor.
```

### op에 인자를 전달했을 때 무엇이 일어나는지에 대한 짧은 설명.

예시:

```markdown
Given a tensor input of numerical type, this operation returns a tensor of
the same type and size with values reversed along dimension `seq_dim`. A
vector `seq_lengths` determines which elements are reversed for each index
within dimension 0 (usually the batch dimension).
```

```markdown
This operation returns a tensor of type `dtype` and dimensions `shape`, with
all elements set to zero.
```

### op가 어떻게 작동하는지를 보여주는 예시.

`squeeze()` op 에 좋은 수도코드 예시가 있습니다:

    shape(input) => `[1, 2, 1, 3, 1, 1]`
    shape(squeeze(input)) =>  `[2, 3]`

`tile()` op 는 좋은 설명문의 예시를 제공합니다:

    For example, tiling `[a, b, c, d]` by 2 produces
    `[[a, b, c, d], [a, b, c, d]]`.

Python에서 코드 예시를 보여주는 것은 도움이 됩니다.
절대 그것을 C++ Ops 파일에 넣지 마세요. 그리고 Python Ops 문서에 넣는 것도 피하세요. 
Ops 생성자가 호출되는 모듈이나 클래스의 docstring에 삽입하세요. 

여기 `image_ops.py` 에 있는 모듈 docstring 예제가 있습니다:

    Tensorflow can convert between images in RGB or HSV. The conversion
    functions work only on `float` images, so you need to convert images in
    other formats using [`convert_image_dtype`](#convert-image-dtype).

    Example:

    ```python
    # Decode an image and convert it to HSV.
    rgb_image = tf.image.decode_png(...,  channels=3)
    rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
    hsv_image = tf.image.rgb_to_hsv(rgb_image)
    ```

### 필요조건, 경고, 중요한 사항들.

예시:

```markdown
This operation requires that: `-1-input.dims() <= dim <= input.dims()`
```

```
Note: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.
```

### 인자와 출력(반환) tensors에 관한 설명.

설명은 요점만 간단히 하세요. 인자 부분에서는 어떻게 실행되는지를 설명할 필요가 없습니다. 

op가 입력 혹은 출력 tensors의 dimension(차원)에 강한 제한이 있으면 언급해야 합니다. 
C++ Ops는 tensor의 타입이 자동으로 "A ..type.. Tensor" 또는 "A Tensor with type in {...list of types...}" 
형태로 더해집니다. 이런 경우에, Op가 차원에 제한이 있으면 "Must be 4-D" 라는 글을 더하거나 
설명의 처음에 `=` 를 추가하고(tensor의 타입이 자동으로 추가되는 것을 막기 위해) "A 4-D float tensor" 와 같이 적으세요.

예를 들어, 여기에 C++ op의 이미지 인자를 문서화하는 두 가지 방법이 있습니다 ("=" 기호를 주목):

```markdown
image: Must be 4-D. The image to resize.
```

```markdown
image:= A 4-D `float` tensor. The image to resize.
```

문서에서 이와 같은 마크다운으로 렌더링됩니다

```markdown
image: A `float` Tensor. Must be 4-D. The image to resize.
```

```markdown
image: A 4-D `float` Tensor. The image to resize.
```

### 선택적인 인자 설명 ("attrs")

doc generator(문서 생성기)는 항상 attrs의 타입과 default 값을 서술합니다. 
C++와 Python doc generator에서 생성되는 설명이 매우 다르기 때문에 equal sign( = 기호)으로 override(오버라이드) 할 수 없습니다.

타입과 default 값 뒤에 흐름이 잘 이어지도록 추가적인 attr 설명을 표현하세요.

`image_ops.py` 에 있는 예시입니다:

```c++
REGISTER_OP("PngDecode")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .Doc(R"doc(
Decode a PNG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.

...

contents: 0-D. The PNG-encoded image.
channels: Number of color channels for the decoded image.
image: 3-D with shape `[height, width, channels]`.
)doc");
```

이것은 아래와 같은 "Args" 부분을 만들어 냅니다:

```markdown
  contents: A string Tensor. 0-D. The PNG-encoded image.
  channels: An optional `int`. Defaults to 0. Number of color channels for the
    decoded image.
  name: A name for the operation (optional).
```




