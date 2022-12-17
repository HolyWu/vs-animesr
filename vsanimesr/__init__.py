from __future__ import annotations

import os

import numpy as np
import tensorrt
import torch
import torch.nn.functional as F
import vapoursynth as vs
from functorch.compile import memory_efficient_fusion
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision

from .vsr_arch import MSRSWVSR

__version__ = '1.0.0'

package_dir = os.path.dirname(os.path.realpath(__file__))


@torch.inference_mode()
def animesr(
    clip: vs.VideoNode,
    device_index: int | None = None,
    nvfuser: bool = False,
    cuda_graphs: bool = False,
    trt: bool = False,
    trt_max_workspace_size: int = 1 << 30,
    trt_cache_path: str = package_dir,
    model: int = 0,
) -> vs.VideoNode:
    """Learning Real-World Super-Resolution Models for Animation Videos

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param nvfuser:                 Enable fusion through nvFuser. Not allowed in TensorRT. (experimental)
    :param cuda_graphs:             Use CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
                                    sequentially. Not allowed in TensorRT.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_max_workspace_size:  Maximum workspace size for TensorRT engine.
    :param trt_cache_path:          Path for TensorRT engine file. Engine will be cached when it's built for the first
                                    time. Note each engine is created for specific settings such as model path/name,
                                    precision, workspace etc, and specific GPUs and it's not portable.
    :param model:                   Model to use.
                                    0 = AnimeSR_v1-PaperModel
                                    1 = AnimeSR_v2
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('animesr: this is not a clip')

    if clip.format.id not in (vs.RGBH, vs.RGBS):
        raise vs.Error('animesr: only RGBH and RGBS formats are supported')

    if not torch.cuda.is_available():
        raise vs.Error('animesr: CUDA is not available')

    if trt:
        if nvfuser:
            raise vs.Error('animesr: nvfuser and trt are mutually exclusive')

        if cuda_graphs:
            raise vs.Error('animesr: cuda_graphs and trt are mutually exclusive')

    if model not in (0, 1):
        raise vs.Error('animesr: model must be 0 or 1')

    fp16 = clip.format.bits_per_sample == 16
    if fp16:
        torch.set_default_tensor_type(torch.HalfTensor)

    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device('cuda', device_index)

    match model:
        case 0:
            model_name = 'AnimeSR_v1-PaperModel.pth'
        case 1:
            model_name = 'AnimeSR_v2.pth'

    model_path = os.path.join(package_dir, model_name)

    scale = modulo = 4

    module = MSRSWVSR(netscale=scale)
    module.load_state_dict(torch.load(model_path, map_location='cpu'))
    module.eval().to(device, memory_format=torch.channels_last)

    pad_w = ((clip.width - 1) // modulo + 1) * modulo
    pad_h = ((clip.height - 1) // modulo + 1) * modulo

    if nvfuser:
        module = memory_efficient_fusion(module)

    if cuda_graphs:
        static_input = torch.empty(1, 9, pad_h, pad_w, device=device, memory_format=torch.channels_last)
        static_sr_in = torch.empty(1, 3, pad_h * scale, pad_w * scale, device=device, memory_format=torch.channels_last)
        static_state_in = torch.empty(1, 64, pad_h, pad_w, device=device, memory_format=torch.channels_last)

        torch.cuda.synchronize(device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(stream):
            module(static_input, static_sr_in, static_state_in)
        torch.cuda.current_stream(device=device).wait_stream(stream)
        torch.cuda.synchronize(device=device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            static_sr_out, static_state_out = module(static_input, static_sr_in, static_state_in)
    elif trt:
        device_name = torch.cuda.get_device_name(device)
        trt_version = tensorrt.__version__
        dimensions = f'{pad_w}x{pad_h}'
        precision = 'fp16' if fp16 else 'fp32'
        trt_engine_path = os.path.join(
            trt_cache_path,
            (
                f'{model_name}'
                + f'_{device_name}'
                + f'_trt-{trt_version}'
                + f'_{dimensions}'
                + f'_{precision}'
                + f'_workspace-{trt_max_workspace_size}'
                + '.pt'
            ),
        )

        if not os.path.isfile(trt_engine_path):
            lower_setting = LowerSetting(
                lower_precision=LowerPrecision.FP16 if fp16 else LowerPrecision.FP32,
                min_acc_module_size=1,
                max_workspace_size=trt_max_workspace_size,
                dynamic_batch=False,
                tactic_sources=1 << int(tensorrt.TacticSource.EDGE_MASK_CONVOLUTIONS)
                | 1 << int(tensorrt.TacticSource.JIT_CONVOLUTIONS),
            )
            lowerer = Lowerer.create(lower_setting=lower_setting)
            module = lowerer(
                module,
                [
                    torch.empty(1, 9, pad_h, pad_w, device=device, memory_format=torch.channels_last),
                    torch.empty(1, 3, pad_h * scale, pad_w * scale, device=device, memory_format=torch.channels_last),
                    torch.empty(1, 64, pad_h, pad_w, device=device, memory_format=torch.channels_last),
                ],
            )
            torch.save(module, trt_engine_path)

        del module
        torch.cuda.empty_cache()
        module = torch.load(trt_engine_path)

    sr = torch.zeros(1, 3, pad_h * scale, pad_w * scale, device=device).to(memory_format=torch.channels_last)
    state = torch.zeros(1, 64, pad_h, pad_w, device=device).to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal sr, state

        img = torch.cat([frame_to_tensor(f[i], device) for i in range(3)], dim=1)

        h, w = img.shape[2:]
        img = F.pad(img, (0, pad_w - w, 0, pad_h - h), 'reflect')

        if cuda_graphs:
            static_input.copy_(img)
            static_sr_in.copy_(sr)
            static_state_in.copy_(state)
            graph.replay()
            sr = static_sr_out
            state = static_state_out
        else:
            sr, state = module(img, sr, state)

        return tensor_to_frame(sr[:, :, : h * scale, : w * scale], f[3].copy())

    clip_prev = clip.std.DuplicateFrames(frames=0).std.Trim(last=clip.num_frames - 1)
    clip_next = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)

    return new_clip.std.ModifyFrame([clip_prev, clip, clip_next, new_clip], inference)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame
