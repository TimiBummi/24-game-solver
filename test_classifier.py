"""
Diagnostic script: test rank_classifier.tflite with different normalizations.
Run in the same environment as the notebook (or locally with tflite-runtime).

Usage:
    python test_classifier.py --model path/to/rank_classifier.tflite \
                              --data  path/to/data/rank_cards
"""

import argparse
import os
import numpy as np

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    load_img = tf.keras.utils.load_img
    img_to_array = tf.keras.utils.img_to_array
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    from PIL import Image
    def load_img(path, target_size):
        return Image.open(path).resize((target_size[1], target_size[0]))
    def img_to_array(img):
        return np.array(img, dtype=np.float32)

IMG_SIZE = 128
CLASS_NAMES = sorted(['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q'])


def run_inference(interpreter, img_float32):
    """Run TFLite inference. img_float32: numpy array [1, H, W, 3]."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_float32)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]  # [13]


def load_model(model_path):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    d = interp.get_input_details()[0]
    print(f"Input:  shape={d['shape']}, dtype={d['dtype']}")
    d = interp.get_output_details()[0]
    print(f"Output: shape={d['shape']}, dtype={d['dtype']}")
    return interp


def test_normalizations(interpreter, img_raw_uint8):
    """Test the same image with different normalizations."""
    print("\n--- Normalization comparison ---")

    # 1. [0, 1]  (notebook val_ds style)
    img_01 = img_raw_uint8.astype(np.float32) / 255.0
    probs_01 = run_inference(interpreter, img_01[np.newaxis])
    pred_01 = CLASS_NAMES[np.argmax(probs_01)]
    print(f"[0,1]   → pred={pred_01:>2}  conf={probs_01.max():.3f}  "
          f"probs={np.round(probs_01, 3)}")

    # 2. [0, 255] (what current Dart code sends)
    img_255 = img_raw_uint8.astype(np.float32)
    probs_255 = run_inference(interpreter, img_255[np.newaxis])
    pred_255 = CLASS_NAMES[np.argmax(probs_255)]
    print(f"[0,255] → pred={pred_255:>2}  conf={probs_255.max():.3f}  "
          f"probs={np.round(probs_255, 3)}")

    # 3. ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_in = (img_raw_uint8.astype(np.float32) / 255.0 - mean) / std
    probs_in = run_inference(interpreter, img_in[np.newaxis])
    pred_in = CLASS_NAMES[np.argmax(probs_in)]
    print(f"ImgNet  → pred={pred_in:>2}  conf={probs_in.max():.3f}  "
          f"probs={np.round(probs_in, 3)}")


def test_accuracy(interpreter, data_dir, normalization='01', n_per_class=5):
    """Test accuracy over the dataset with given normalization."""
    print(f"\n--- Accuracy test (normalization={normalization}, "
          f"n_per_class={n_per_class}) ---")

    correct = 0
    total = 0

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: class dir not found: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:n_per_class]

        true_idx = CLASS_NAMES.index(class_name)

        for fname in files:
            path = os.path.join(class_dir, fname)
            img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            arr = img_to_array(img)  # [H, W, 3] float32 in [0,255]

            if normalization == '01':
                inp = (arr / 255.0)[np.newaxis]
            elif normalization == '255':
                inp = arr[np.newaxis]
            elif normalization == 'imagenet':
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                inp = ((arr / 255.0 - mean) / std)[np.newaxis]

            probs = run_inference(interpreter, inp.astype(np.float32))
            pred_idx = int(np.argmax(probs))
            match = pred_idx == true_idx
            correct += match
            total += 1

    acc = correct / total * 100 if total > 0 else 0
    print(f"  Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to rank_classifier.tflite')
    parser.add_argument('--data',  default=None,  help='Path to rank_cards directory')
    parser.add_argument('--image', default=None,  help='Single test image path')
    args = parser.parse_args()

    interp = load_model(args.model)

    if args.image:
        img = load_img(args.image, target_size=(IMG_SIZE, IMG_SIZE))
        arr = img_to_array(img)
        test_normalizations(interp, arr)

    if args.data:
        # Test one sample image from each class for normalization comparison
        first_class = CLASS_NAMES[0]
        sample_dir = os.path.join(args.data, first_class)
        if os.path.isdir(sample_dir):
            files = [f for f in os.listdir(sample_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                sample_path = os.path.join(sample_dir, files[0])
                img = load_img(sample_path, target_size=(IMG_SIZE, IMG_SIZE))
                arr = img_to_array(img)
                test_normalizations(interp, arr)

        for norm in ['01', '255', 'imagenet']:
            test_accuracy(interp, args.data, normalization=norm, n_per_class=10)


if __name__ == '__main__':
    main()
