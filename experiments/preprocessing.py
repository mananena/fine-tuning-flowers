from PIL import Image, ImageOps
import numpy as np

def preprocess(image, spec):
    """Одинаковое преобразование для оценки PyTorch и приложения ONNX."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    size = spec['size']
    resize = round(size / spec['crop_pct'])
    w, h = image.size
    if w <= h:
        shape = (resize, round(h * resize / w))
    else:
        shape = (round(w * resize / h), resize)
    image = image.resize(shape, Image.Resampling.BICUBIC)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(spec['mean'], dtype=np.float32)) / np.asarray(spec['std'], dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
