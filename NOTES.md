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