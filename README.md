# AnimeSR
Learning Real-World Super-Resolution Models for Animation Videos, based on https://github.com/TencentARC/AnimeSR.

Note that the SR result and hidden state of the current frame will propagate to the next frame. Hence you may not get the same result if you jump to random frames in the editor/previewer instead of rendering linearly.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.5.0.dev or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional Python packages:
- [TensorRT](https://developer.nvidia.com/tensorrt/) 10.0.1
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.5.0.dev

To install TensorRT, run `pip install tensorrt==10.0.1 tensorrt-cu12_bindings==10.0.1 tensorrt-cu12_libs==10.0.1 --extra-index-url https://pypi.nvidia.com`

To install Torch-TensorRT, run `pip install --pre torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124`


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
