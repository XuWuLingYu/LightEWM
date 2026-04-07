import math
import numpy as np
import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image
try:
    import torchaudio
except Exception:
    torchaudio = None


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data):
        if isinstance(data, dict):
            data = data["path"]
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, resize_mode="crop"):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.resize_mode = resize_mode

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        if self.resize_mode == "stretch":
            return torchvision.transforms.functional.resize(
                image,
                (target_height, target_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
        if self.resize_mode == "letterbox":
            scale = min(target_width / width, target_height / height)
            resized_w = max(1, round(width * scale))
            resized_h = max(1, round(height * scale))
            resized = torchvision.transforms.functional.resize(
                image,
                (resized_h, resized_w),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
            offset_x = (target_width - resized_w) // 2
            offset_y = (target_height - resized_h) // 2
            canvas.paste(resized, (offset_x, offset_y))
            return canvas
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_rate=24,
        fix_frame_rate=False,
        video_sampling_mode="prefix",
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate
        self.video_sampling_mode = video_sampling_mode

    def get_reader(self, data: str):
        return imageio.get_reader(data)

    def safe_count_frames(self, reader):
        try:
            return int(reader.count_frames())
        except Exception:
            pass
        try:
            length = reader.get_length()
            if length is not None and length != float("inf") and length >= 0:
                return int(length)
        except Exception:
            pass
        return 0

    def get_available_num_frames(self, reader):
        total_original_frames = self.safe_count_frames(reader)
        if not self.fix_frame_rate:
            return total_original_frames
        meta_data = reader.get_meta_data()
        if "duration" in meta_data and meta_data["duration"] is not None:
            duration = meta_data["duration"]
        else:
            raw_fps = meta_data.get("fps", None)
            duration = total_original_frames / raw_fps if raw_fps else 0
        total_available_frames = math.floor(duration * self.frame_rate)
        return int(total_available_frames)

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        total_frames = self.get_available_num_frames(reader)
        if int(total_frames) < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int) -> int:
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))        
        frame_id = min(frame_id, total_raw_frames - 1)
        return frame_id


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        frame_rate=24,
        fix_frame_rate=False,
        video_sampling_mode="prefix",
    ):
        FrameSamplerByRateMixin.__init__(
            self,
            num_frames,
            time_division_factor,
            time_division_remainder,
            frame_rate,
            fix_frame_rate,
            video_sampling_mode=video_sampling_mode,
        )
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def parse_video_item(self, data):
        if isinstance(data, dict):
            return (
                data["path"],
                int(data.get("context_start", 0)),
                data.get("context_window_size", None),
                bool(data.get("pad_last", False)),
            )
        return data, 0, None, False

    def build_sequence_ids(self, available_frames, context_start, context_window_size, pad_last):
        if context_window_size is None:
            if self.video_sampling_mode == "uniform_full_video":
                if available_frames <= 0:
                    return []
                if available_frames == 1:
                    return [0] * self.num_frames
                return np.rint(np.linspace(0, available_frames - 1, self.num_frames)).astype(int).tolist()
            num_frames = min(self.num_frames, available_frames)
            if num_frames < self.num_frames:
                while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                    num_frames -= 1
            return list(range(max(0, num_frames)))
        start = max(0, context_start)
        size = int(context_window_size)
        if available_frames <= 0:
            return []
        if start >= available_frames:
            start = available_frames - 1
        end = start + size
        if end <= available_frames:
            return list(range(start, end))
        valid = list(range(start, available_frames))
        if pad_last:
            valid.extend([available_frames - 1] * (end - available_frames))
            return valid
        fallback_start = max(0, available_frames - size)
        return list(range(fallback_start, available_frames))

    def __call__(self, data):
        data, context_start, context_window_size, pad_last = self.parse_video_item(data)
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        available_frames = self.get_available_num_frames(reader)
        total_raw_frames = max(1, self.safe_count_frames(reader))
        seq_frame_ids = self.build_sequence_ids(available_frames, context_start, context_window_size, pad_last)
        frames = []
        for frame_id in seq_frame_ids:
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data):
        context_start = 0
        context_window_size = None
        pad_last = False
        if isinstance(data, dict):
            context_start = int(data.get("context_start", 0))
            context_window_size = data.get("context_window_size", None)
            pad_last = bool(data.get("pad_last", False))
            data = data["path"]

        frames = []
        images = iio.imread(data, mode="RGB")
        if context_window_size is None:
            num_frames = self.get_num_frames(data)
            frame_ids = list(range(min(num_frames, len(images))))
        else:
            available_frames = len(images)
            if available_frames <= 0:
                frame_ids = []
            else:
                start = max(0, context_start)
                size = int(context_window_size)
                if start >= available_frames:
                    start = available_frames - 1
                end = start + size
                if end <= available_frames:
                    frame_ids = list(range(start, end))
                else:
                    frame_ids = list(range(start, available_frames))
                    if pad_last:
                        frame_ids.extend([available_frames - 1] * (end - available_frames))
                    else:
                        fallback_start = max(0, available_frames - size)
                        frame_ids = list(range(fallback_start, available_frames))

        for frame_id in frame_ids:
            img = images[frame_id]
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        path = data["path"] if isinstance(data, dict) else data
        file_ext_name = path.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        if isinstance(data, dict):
            data = data.copy()
            data["path"] = os.path.join(self.base_path, data["path"])
            return data
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)

    def __call__(self, data: str):
        if torchaudio is None:
            raise RuntimeError(
                "torchaudio is not available. Please install a torchaudio build compatible "
                "with your current torch/CUDA runtime."
            )
        reader = self.get_reader(data)
        num_frames = self.get_num_frames(reader)
        duration = num_frames / self.frame_rate
        waveform, sample_rate = torchaudio.load(data)
        target_samples = int(duration * sample_rate)
        current_samples = waveform.shape[-1]
        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform, sample_rate
