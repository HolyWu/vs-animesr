from __future__ import annotations

import math
import os
import warnings
from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .vsr_arch import MSRSWVSR

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "At pre-dispatch tracing")
warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class AnimeSRModel(IntEnum):
    AnimeSR_v1 = 0
    AnimeSR_v2 = 1


@torch.inference_mode()
def animesr(
    clip: vs.VideoNode,
    device_index: int = 0,
    model: AnimeSRModel = AnimeSRModel.AnimeSR_v2,
    trt: bool = False,
    trt_debug: bool = False,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [640, 360],
    trt_max_shape: list[int] = [1280, 720],
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Learning Real-World Super-Resolution Models for Animation Videos

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param model:                   Model to use.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_min_shape:           Min size of dynamic shapes.
    :param trt_opt_shape:           Opt size of dynamic shapes.
    :param trt_max_shape:           Max size of dynamic shapes.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("animesr: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("animesr: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("animesr: CUDA is not available")

    if model not in AnimeSRModel:
        raise vs.Error("animesr: model must be one of the members in AnimeSRModel")

    if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
        raise vs.Error("animesr: trt_min_shape must be a list with 2 items")

    if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
        raise vs.Error("animesr: trt_opt_shape must be a list with 2 items")

    if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
        raise vs.Error("animesr: trt_max_shape must be a list with 2 items")

    if any(trt_min_shape[i] >= trt_max_shape[i] for i in range(2)):
        raise vs.Error("animesr: trt_min_shape must be less than trt_max_shape")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    model_name = f"{AnimeSRModel(model).name}.pth"

    state_dict = torch.load(os.path.join(model_dir, model_name), map_location=device, weights_only=True, mmap=True)

    with torch.device("meta"):
        module = MSRSWVSR()
    module.load_state_dict(state_dict, assign=True)
    module.eval().to(device)
    if fp16:
        module.half()

    w = clip.width
    h = clip.height
    pad_w = math.ceil(w / 4) * 4
    pad_h = math.ceil(h / 4) * 4
    padding = (0, pad_w - w, 0, pad_h - h)

    if trt:
        import tensorrt
        import torch_tensorrt

        for i in range(2):
            trt_min_shape[i] = math.ceil(max(trt_min_shape[i], 1) / 4) * 4
            trt_opt_shape[i] = math.ceil(max(trt_opt_shape[i], 1) / 4) * 4
            trt_max_shape[i] = math.ceil(max(trt_max_shape[i], 1) / 4) * 4

        dimensions = (
            f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
            f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
            f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
        )

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{dimensions}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            trt_min_shape.reverse()
            trt_opt_shape.reverse()
            trt_max_shape.reverse()

            trt_min_shape_out = [trt_min_shape[i] * 4 for i in range(2)]
            trt_opt_shape_out = [trt_opt_shape[i] * 4 for i in range(2)]
            trt_max_shape_out = [trt_max_shape[i] * 4 for i in range(2)]

            example_tensors = (
                torch.zeros((1, 9, pad_h, pad_w), dtype=dtype, device=device),
                torch.zeros((1, 3, pad_h * 4, pad_w * 4), dtype=dtype, device=device),
                torch.zeros((1, 64, pad_h, pad_w), dtype=dtype, device=device),
            )

            _height = torch.export.Dim("height", min=trt_min_shape[0] // 4, max=trt_max_shape[0] // 4)
            _width = torch.export.Dim("width", min=trt_min_shape[1] // 4, max=trt_max_shape[1] // 4)
            dim_height = _height * 4
            dim_width = _width * 4
            dim_height_out = dim_height * 4
            dim_width_out = dim_width * 4
            dynamic_shapes = {
                "x": {2: dim_height, 3: dim_width},
                "fb": {2: dim_height_out, 3: dim_width_out},
                "state": {2: dim_height, 3: dim_width},
            }

            exported_program = torch.export.export(module, example_tensors, dynamic_shapes=dynamic_shapes)

            inputs = [
                torch_tensorrt.Input(
                    min_shape=[1, 9] + trt_min_shape,
                    opt_shape=[1, 9] + trt_opt_shape,
                    max_shape=[1, 9] + trt_max_shape,
                    dtype=dtype,
                    name="x",
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 3] + trt_min_shape_out,
                    opt_shape=[1, 3] + trt_opt_shape_out,
                    max_shape=[1, 3] + trt_max_shape_out,
                    dtype=dtype,
                    name="fb",
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 64] + trt_min_shape,
                    opt_shape=[1, 64] + trt_opt_shape,
                    max_shape=[1, 64] + trt_max_shape,
                    dtype=dtype,
                    name="state",
                ),
            ]

            module = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs,
                enabled_precisions={dtype},
                debug=trt_debug,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
                device=device,
                assume_dynamic_shape_support=True,
            )

            torch_tensorrt.save(module, trt_engine_path, output_format="torchscript", inputs=example_tensors)

        module = torch.jit.load(trt_engine_path).eval()

    out = torch.zeros((1, 3, pad_h * 4, pad_w * 4), dtype=dtype, device=device)
    state = torch.zeros((1, 64, pad_h, pad_w), dtype=dtype, device=device)

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal out, state

        img = torch.cat([frame_to_tensor(f[i], device) for i in range(3)], dim=1)
        img = F.pad(img, padding, "replicate")

        out, state = module(img, out, state)
        return tensor_to_frame(out[:, :, : h * 4, : w * 4], f[3].copy())

    clip_prev = clip.std.DuplicateFrames(frames=0).std.Trim(last=clip.num_frames - 1)
    clip_next = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    new_clip = clip.std.BlankClip(width=clip.width * 4, height=clip.height * 4, keep=True)

    return new_clip.std.ModifyFrame([clip_prev, clip, clip_next, new_clip], inference)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return (
        torch.stack([torch.from_numpy(np.asarray(frame[plane])).to(device) for plane in range(frame.format.num_planes)])
        .unsqueeze(0)
        .clamp(0.0, 1.0)
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane])
    return frame
