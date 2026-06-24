import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATHS = [
    r'mobile_accuracy_optimization\\yolo_runs\\yolo26n-cls\\weights\\best_saved_model\\best_float32.tflite',
    r'LiteRT-for-Android\\app\\src\\main\\assets\\yolo26n_cls_best_float32.tflite',
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
    layout, height, width, channels = resolve_input_layout(input_shape)
    image = Image.open(image_path).convert('RGB').resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.shape[-1] != channels:
        raise ValueError(f'Unexpected channel count: {arr.shape}')
    if layout == 'NCHW':
        arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    if np.issubdtype(input_dtype, np.integer):
        # Quantized input is not expected for this model, but handle it safely.
        scale, zero_point = quantization
        arr = arr / scale + zero_point
        arr = np.clip(np.rint(arr), np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
    else:
        arr = arr.astype(input_dtype)
    return arr


def resolve_input_layout(input_shape):
    shape = [int(dim) for dim in input_shape]
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f'Expected a batched 4D input tensor, got {shape}')
    if shape[-1] == 3:
        return 'NHWC', shape[1], shape[2], shape[3]
    if shape[1] == 3:
        return 'NCHW', shape[2], shape[3], shape[1]
    raise ValueError(f'Cannot infer image layout from input shape: {shape}')


def to_probabilities(output):
    values = output.astype(np.float32).reshape(-1)
    total = float(values.sum())
    if np.all(values >= 0.0) and np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
        return values
    exp_values = np.exp(values - np.max(values))
    return exp_values / exp_values.sum()


if __name__ == '__main__':
    image_paths = find_one_image_per_class(TEST_ROOT)
    if len(image_paths) != len(CLASS_NAMES):
        raise SystemExit(f'Expected one image for each class, found {len(image_paths)}: {image_paths}')

    print('Selected test images:')
    for path in image_paths:
        print('  ', path)
    print()

    for model in MODEL_PATHS:
        if not os.path.exists(model):
            print('SKIP missing model:', model)
            continue
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
            probs = to_probabilities(out)
            print(f' Image class={label}')
            for i, class_name in enumerate(CLASS_NAMES):
                print(f'   {class_name}: {probs[i]:.6f}')
            print('   top', CLASS_NAMES[int(np.argmax(probs))], 'score', float(np.max(probs)))
        print()
