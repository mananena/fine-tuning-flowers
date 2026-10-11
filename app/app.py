import json
from pathlib import Path
import time
import gradio as gr
import numpy as np
import onnxruntime as ort
from experiments.preprocessing import preprocess

MODELS = Path(__file__).resolve().parents[1] / 'experiments/models'

def build_app():
    if not (MODELS/'best.onnx').exists():
        raise SystemExit('Сначала выполните обучение и python -m experiments.train --finalize')
    meta = json.loads((MODELS/'metadata.json').read_text(encoding='utf-8'))
    session = ort.InferenceSession(str(MODELS/'best.onnx'), providers=['CPUExecutionProvider'])
    def predict(image):
        if image is None:
            raise gr.Error('Загрузите фотографию.')
        x = preprocess(image, meta['preprocessing'])[None]
        start = time.perf_counter()
        logits = session.run(None, {'image': x})[0][0]
        elapsed = (time.perf_counter() - start) * 1000
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return dict(zip(meta['classes'], map(float, probs))), f'Инференс на CPU: {elapsed:.1f} мс'
    return gr.Interface(predict, inputs=gr.Image(type='pil', label='Фотография'),
                 outputs=[gr.Label(label='Результат'), gr.Textbox(label='Время')],
                 title='Классификация цветов', description='Загрузите фотографию маргаритки, розы или подсолнуха.')

def main():
    build_app().launch(share=False)

if __name__ == '__main__':
    main()
