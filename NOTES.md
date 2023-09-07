# Steps

1. need to create an env on python 3.6

2. modify the `setup.py` file:

replace Line 10:
```python
with open("README.md") as f:
```

by:
```python
with open("README.md", encoding="utf-8") as f:
```
Since the default encoding on windows is not utf-8, it will cause an error when you try to install the package:

```bash
Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-req-build-r8qya3m3/setup.py", line 11, in <module>
        readme = f.read()
      File "/home/kaixuan_cuda11/anaconda3/envs/DPR/lib/python3.6/encodings/ascii.py", line 26, in decode
        return codecs.ascii_decode(input, self.errors)[0]
    UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 486: ordinal not in range(128)
    ----------------------------------------
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.`
```

3. install the package on the device, since `faiss-cpu` needs the `swig` tool, you need to install it first by running:

On Ubuntu:

```bash
sudo apt-get install swig
```
On Mac:

```bash
brew install swig
```


> SWIG is a software development tool that connects programs written in C and C++ with various high-level programming languages, and it seems that faiss-cpu requires it for building.


Otherwise, you will get an error like this:

```bash
error: command 'swig' failed with exit status 1
```

4. However, the version of `faiss-cpu` is not specified in the `setup.py` file, and the latest version of `faiss-cpu` is not compatible with the latest version of `torch`, so we need to install the specified version of `faiss-cpu` or install a compatible version of `torch` by running:

```bash
conda install faiss-cpu -c pytorch
```

5. Install the left packages by rerunning:

> :warning It would still encountered this error since `pip` still tried to install `faiss-cpu`, although it has been installed by `conda` before. So we need to install the left packages by commenting the `faiss-cpu` in the `setup.py` file.


Then we can install the left packages by rerunning:
```bash
pip install .
```

6. Install the missing packages `distributed_faiss` within the `dense_retriever.py` file:

:warning: This step aims to solve the following errors:

```bash
Import "distributed_faiss.client" could not be resolved Pylance(reportMissinglmports) [Ln 194, Col 14]
Import "distributed_faiss.index_cfg" could not be resolved Pylance(reportMissinglmports) [Ln 206, Col 14]
Import "distributed_faiss.index_cfg" could not be resolved Pylance(reportMissinglmports) [Ln 231, Col 14]
Import "distributed_faiss.index_state" could not be resolved Pylance(reportMissinglmports) [Ln 290, Col 14]
```
---

(1) The package `distributed_faiss` seems not available on `pip`, but we found it on `GitHub`.

```bash
git clone https://github.com/facebookresearch/distributed-faiss.git distributed_faiss
```
> :warning: We need to rename the folder `distributed_faiss` to `distributed_faiss` since we would import it by this name.
(`-` is not allowed in the name of the package.)


(2) Install and test it accroding to the README file in it:

```bash
pip install -e .
python -m unittest discover tests
```

(3) Then we need to replace the lines that import `distributed_faiss` in the `dense_retriever.py` file by
replace `distributed_faiss.XXX` as `distributed_faiss.distributed_faiss.XXX`.

> ? Since I did not find a better way to call the lib although we installed it in (2), I just replace it by this way. :(

7. Solve the import error of `apex`:

Since `apex` is deprecated: https://github.com/NVIDIA/apex/issues/1214, we need to replace the `apex` by `torch.cuda.amp`.



Replace Line 502 in `train_dense_encoder.py` and Line 315 in `train_extractive_reader.py`:

```python
from apex import amp
```

by 

```python
from torch.cuda import amp
```
