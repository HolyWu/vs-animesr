# AnimeSR
Learning Real-World Super-Resolution Models for Animation Videos, based on https://github.com/TencentARC/AnimeSR.

Note that the SR result and hidden state of the current frame will propagate to the next frame. Hence you may not get the same result if you jump to random frames in the editor/previewer instead of rendering linearly.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13
- [VapourSynth](http://www.vapoursynth.com/) R55+

`trt` requires additional runtime libraries:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 11.7
- [cuDNN](https://developer.nvidia.com/cudnn) 8.6
- [TensorRT](https://developer.nvidia.com/tensorrt) 8.5

For ease of installation on Windows, you can download the 7z file on [Releases](https://github.com/HolyWu/vs-animesr/releases) which contains required runtime libraries and Python wheel file. Either add the unzipped directory to your system `PATH` or copy the DLL files to a directory which is already in your system `PATH`. Finally pip install the Python wheel file.


## Installation
```
pip install -U vsanimesr
```


## Usage
```python
from vsanimesr import animesr

ret = animesr(clip)
```

See `__init__.py` for the description of the parameters.
