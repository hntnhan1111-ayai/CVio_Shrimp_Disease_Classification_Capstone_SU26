import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATHS = [
    r'mobile_accuracy_optimization\\yolo_runs\\yolo26n-cls\\weights\\best_saved_model\\best_float32.tflite',
    r'LiteRT-for-Android\\app\\src\\main\\assets\\best_float32.tflite',
]
TEST_ROOT = r'mobile_accuracy_optimization\\yolo_dataset\\test'
CLASS_NAMES = ['Healthy', 'BG', 'WSSV', 'WSSV_BG']


def find_one_image_per_class(root):
    images = {}
    for dirpath, dirnames, filenames in os.walk(root):
        label = os.path.basename(dirpath).split('. ', 1)[-1]
        if label not in CLASS_NAMES:
            continue
        for filename in sorted(filenames):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                images[label] = os.path.join(dirpath, filename)
                break
    return [images[label] for label in CLASS_NAMES if label in images]


def preprocess_yolo(image_path, input_shape, input_dtype, quantization):
    _, height, width, channels = input_shape
    image = Image.open(image_path).convert('RGB').resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.shape[-1] != channels:
        raise ValueError(f'Unexpected channel count: {arr.shape}')
    arr = np.expand_dims(arr, axis=0)
    if np.issubdtype(input_dtype, np.integer):
        # Quantized input is not expected for this model, but handle it safely.
        scale, zero_point = quantization
        arr = arr / scale + zero_point
        arr = np.clip(np.rint(arr), np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
    else:
        arr = arr.astype(input_dtype)
    return arr


if __name__ == '__main__':
    image_paths = find_one_image_per_class(TEST_ROOT)
    if len(image_paths) != len(CLASS_NAMES):
        raise SystemExit(f'Expected one image for each class, found {len(image_paths)}: {image_paths}')

    print('Selected test images:')
    for path in image_paths:
        print('  ', path)
    print()

    for model in MODEL_PATHS:
        print('MODEL:', model)
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        print(' input shape', input_details['shape'], 'dtype', input_details['dtype'], 'quant', input_details.get('quantization', (0.0, 0)))

        for image_path in image_paths:
            label = os.path.basename(os.path.dirname(image_path)).split('. ', 1)[-1]
            inp = preprocess_yolo(image_path, input_details['shape'], input_details['dtype'], input_details.get('quantization', (0.0, 0)))
            interpreter.set_tensor(input_details['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index']).reshape(-1)
            if np.issubdtype(out.dtype, np.integer):
                scale, zp = output_details['quantization']
                out = (out.astype(np.float32) - zp) * scale
            probs = np.exp(out - np.max(out))
            probs = probs / probs.sum()
            print(f' Image class={label}')
            for i, class_name in enumerate(CLASS_NAMES):
                print(f'   {class_name}: {probs[i]:.6f}')
            print('   top', CLASS_NAMES[int(np.argmax(probs))], 'score', float(np.max(probs)))
        print()
