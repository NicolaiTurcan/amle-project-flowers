"""Model code shared by the app: classify a photo and measure how similar it is to the training photos.

Every model in models/ takes the raw photo (pixel values 0 to 255), does its own preprocessing and outputs one
probability per flower. Because a 3-class model always picks one of its three flowers, the app also compares the
photo's features from the pretrained base with the features of the training photos, and answers "no match found"
when the photo is not similar enough.

The features of the training photos are computed once and saved to models/reference_features.npz, so that the app
does not need to run 300 photos through the network on a small server. After training new models, rebuild the file:

    uv run python flower_model.py
"""

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
TRAIN_DIR = ROOT / "data" / "train"
REFERENCE_FILE = MODELS_DIR / "reference_features.npz"

CLASSES = ["rose", "sunflower", "tulip"]  # the folder names, in alphabetical order, as in the training data

# "No match" rules, calibrated on the validation and test photos and on photos of other objects and other flowers:
# - SIMILARITY: mean cosine similarity of the photo's MobileNetV2 features to its 5 closest training photos.
#   Real roses, sunflowers and tulips score about 0.46-0.75; cars, landscapes, buildings and food score below 0.44.
# - CONFIDENCE: the model's top probability. Below 0.60 the model is not sure which of the three flowers it sees.
MIN_SIMILARITY = 0.44
MIN_CONFIDENCE = 0.60
NEIGHBOURS = 5


def split_layers(model):
    """Split the model into the layers up to the pretrained base (the nested model with a pooled output) and the head."""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.Model) and len(layer.output.shape) == 2:
            return model.layers[: i + 1], model.layers[i + 1 :]
    raise ValueError("The model has no pretrained base with a pooled output.")


def prepare_photo(photo_bytes, size):
    """Decode and resize a photo with TensorFlow, as image_dataset_from_directory did for the training photos."""
    image = tf.io.decode_image(photo_bytes, channels=3, expand_animations=False)
    return tf.image.resize(image, size).numpy()  # bilinear, the default of image_dataset_from_directory


def features_and_probabilities(model, pixels):
    """One pass through the model: L2-normalised base features and class probabilities for a batch of raw photos.

    Keep the batches small: in eager mode TensorFlow needs about 1.5 GB for a batch of 32 photos.
    """
    base_layers, head_layers = split_layers(model)
    x = tf.convert_to_tensor(pixels)
    for layer in base_layers:
        x = layer(x, training=False)  # augmentation and dropout layers do nothing when training=False
    features = x
    for layer in head_layers:
        x = layer(x, training=False)
    features = features.numpy()
    return features / np.linalg.norm(features, axis=1, keepdims=True), x.numpy()


def compute_reference_features(model):
    """Features of every training photo, one photo at a time to keep the memory low."""
    size = model.input_shape[1:3]
    rows = []
    for path in sorted(TRAIN_DIR.glob("*/*.jpg")):
        pixels = prepare_photo(tf.io.read_file(str(path)), size)[np.newaxis]
        rows.append(features_and_probabilities(model, pixels)[0][0])
    return np.stack(rows)


def similarity_to(reference, features):
    """Mean cosine similarity of one photo's features to its closest training photos."""
    similarities = reference.astype(np.float32) @ features
    return float(np.sort(similarities)[-NEIGHBOURS:].mean())


if __name__ == "__main__":
    saved = {}
    for path in sorted(MODELS_DIR.glob("*.keras")):
        saved[path.name] = compute_reference_features(keras.models.load_model(path)).astype(np.float16)
        print(f"{path.name}: {saved[path.name].shape}")
    np.savez_compressed(REFERENCE_FILE, **saved)
    print(f"Saved {REFERENCE_FILE.relative_to(ROOT)} ({REFERENCE_FILE.stat().st_size / 1024:.0f} KB)")
