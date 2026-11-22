"""Flower Classifier: a Streamlit app that tells roses, sunflowers and tulips apart.

Run it from the repository folder: uv run streamlit run app.py

Every model in models/ takes the raw photo (pixel values 0 to 255), does its own preprocessing and outputs one
probability per flower. Because a 3-class model always picks one of its three flowers, the app also checks whether
the photo looks like the training photos at all, and answers "no match found" when it does not.
"""

from pathlib import Path

import keras
import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
REPO_URL = "https://github.com/NicolaiTurcan/amle-project-flowers"

CLASSES = ["rose", "sunflower", "tulip"]  # the folder names, in alphabetical order, as in the training data
FLOWERS = {
    "rose": {"emoji": "🌹", "color": "#d94a6a"},
    "sunflower": {"emoji": "🌻", "color": "#e0a516"},
    "tulip": {"emoji": "🌷", "color": "#c8559c"},
}

# Example photos: three test photos that every model classifies correctly, and one photo that is not a flower
# (by André Spieker on Unsplash, https://unsplash.com/photos/8wTPqxlnKM4, Unsplash License).
EXAMPLES = {
    "🌹 Rose": TEST_DIR / "rose" / "2675221506_5286c0595f.jpg",
    "🌻 Sunflower": TEST_DIR / "sunflower" / "2823659190_afdabee45c.jpg",
    "🌷 Tulip": TEST_DIR / "tulip" / "2503489175_f0848d3e8e.jpg",
    "🐶 Not a flower": ROOT / "assets" / "examples" / "dog.jpg",
}

# Shown next to each model in the selector; accuracies from the notebook (60 validation, 30 test photos).
# The first entry is the recommended model and is selected by default.
MODEL_INFO = {
    "dense_head.keras": {
        "name": "Dense head (recommended)",
        "about": "Frozen MobileNetV2 + a single dense layer, trained without augmentation. "
        "Chosen as the final model for its best validation accuracy.",
        "val": 0.867,
        "test": 0.933,
    },
    "augmented_head.keras": {
        "name": "Augmented head",
        "about": "Frozen MobileNetV2 + dense head, trained with random flips, rotations and zooms.",
        "val": 0.800,
        "test": 0.967,
    },
    "finetuned_model.keras": {
        "name": "Augmented head, extra training",
        "about": "The augmented model trained for 15 more epochs with a 100x lower learning rate.",
        "val": 0.800,
        "test": 0.967,
    },
}

# "No match" rules, calibrated on the validation and test photos and on photos of other objects and other flowers:
# - SIMILARITY: mean cosine similarity of the photo's MobileNetV2 features to its 5 closest training photos.
#   Real roses, sunflowers and tulips score about 0.46-0.75; cars, landscapes, buildings and food score below 0.44.
# - CONFIDENCE: the model's top probability. Below 0.60 the model is not sure which of the three flowers it sees.
MIN_SIMILARITY = 0.44
MIN_CONFIDENCE = 0.60
NEIGHBOURS = 5


# ---------------------------------------------------------------------------------------------------------------------
# Model and features
# ---------------------------------------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)  # load each model only once, not again every time the page changes
def load_model(path):
    return keras.models.load_model(path)


def feature_layers(model):
    """The layers from the input up to the pretrained base (the nested model whose output is a feature vector)."""
    layers = []
    for layer in model.layers:
        layers.append(layer)
        if isinstance(layer, keras.Model) and len(layer.output.shape) == 2:
            return layers
    raise ValueError("The model has no pretrained base with a pooled output.")


def extract_features(model, pixels):
    """L2-normalised features of the pretrained base, for a batch of raw photos."""
    x = tf.convert_to_tensor(pixels)
    for layer in feature_layers(model):
        x = layer(x, training=False)  # augmentation layers do nothing when training=False
    x = x.numpy()
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def prepare_photo(photo_bytes, size):
    """Decode and resize a photo with TensorFlow, as image_dataset_from_directory did for the training photos."""
    image = tf.io.decode_image(photo_bytes, channels=3, expand_animations=False)
    return tf.image.resize(image, size).numpy()  # bilinear, the default of image_dataset_from_directory


@st.cache_resource(show_spinner=False)
def reference_features(model_path):
    """Features of every training photo, to measure how similar a new photo is to what the model has seen."""
    model = load_model(model_path)
    size = model.input_shape[1:3]
    photos = np.stack([prepare_photo(tf.io.read_file(str(p)), size) for p in sorted(TRAIN_DIR.glob("*/*.jpg"))])
    return np.concatenate([extract_features(model, photos[i : i + 32]) for i in range(0, len(photos), 32)])


def classify(model_path, photo_bytes):
    model = load_model(model_path)
    pixels = prepare_photo(photo_bytes, model.input_shape[1:3])[np.newaxis]  # a batch of one photo
    probabilities = model.predict(pixels, verbose=0)[0]
    similarities = extract_features(model, pixels)[0] @ reference_features(model_path).T
    similarity = float(np.sort(similarities)[-NEIGHBOURS:].mean())
    return probabilities, similarity


# ---------------------------------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Flower Classifier", page_icon="🌷", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 840px; }
      .hero { text-align: center; margin-bottom: 1.2rem; }
      .hero h1 { font-size: 2.4rem; margin-bottom: .25rem; }
      .hero p { color: #6b6b7b; font-size: 1.05rem; margin: 0; }
      .chips { display: flex; gap: .5rem; justify-content: center; margin-top: .9rem; flex-wrap: wrap; }
      .chip { background: #ffffff; border: 1px solid #ecdfe6; border-radius: 999px; padding: .3rem .85rem;
              font-size: .92rem; color: #3b3b4f; }
      .chips.left { justify-content: flex-start; }
      .step { font-weight: 700; font-size: 1.1rem; margin: 1.4rem 0 .15rem; color: #2f2f45; }
      .step span { display: inline-block; width: 1.6rem; height: 1.6rem; border-radius: 50%; text-align: center;
                   line-height: 1.6rem; margin-right: .5rem; background: #c8559c; color: white; font-size: .9rem; }
      .hint { color: #6b6b7b; font-size: .93rem; margin-bottom: .6rem; }
      .card { background: #ffffff; border: 1px solid #ecdfe6; border-radius: 12px; padding: .8rem 1rem;
              font-size: .92rem; color: #4a4a5e; }
      .card b { color: #2f2f45; }
      .model-card { margin-top: .4rem; }
      .result { border-radius: 16px; padding: 1.2rem 1.3rem; background: #ffffff; border: 1px solid #ecdfe6; }
      .result .label { font-size: .85rem; text-transform: uppercase; letter-spacing: .06em; color: #8a8a9a; }
      .result .answer { font-size: 1.9rem; font-weight: 800; margin: .1rem 0 .2rem; }
      .result .sub { color: #6b6b7b; font-size: .95rem; margin-bottom: .9rem; }
      .result.nomatch { background: #fff8ec; border-color: #f3d9a4; }
      .bar-row { display: flex; align-items: center; gap: .6rem; margin: .35rem 0; font-size: .93rem; }
      .bar-name { width: 7.2rem; color: #3b3b4f; }
      .bar-track { flex: 1; height: .65rem; background: #f1ebee; border-radius: 999px; overflow: hidden; }
      .bar-fill { height: 100%; border-radius: 999px; }
      .bar-value { width: 3rem; text-align: right; color: #4a4a5e; font-variant-numeric: tabular-nums; }
      .about h3 { font-size: 1.25rem; margin: 1.8rem 0 .5rem; color: #2f2f45; }
      .about p, .about li { color: #4a4a5e; font-size: .96rem; line-height: 1.6; }
      .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin: .6rem 0; }
      .stat { background: #ffffff; border: 1px solid #ecdfe6; border-radius: 12px; padding: .8rem 1rem; }
      .stat .value { font-size: 1.6rem; font-weight: 800; color: #c8559c; }
      .stat .name { font-size: .85rem; color: #6b6b7b; }
      .flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .6rem; margin: .6rem 0; }
      .flow .card { position: relative; padding-top: 1.6rem; }
      .flow .num { position: absolute; top: .55rem; left: 1rem; font-size: .75rem; font-weight: 700; color: #c8559c;
                   letter-spacing: .05em; }
      .table { width: 100%; border-collapse: collapse; font-size: .93rem; background: #ffffff; border-radius: 12px;
               overflow: hidden; border: 1px solid #ecdfe6; }
      .table th, .table td { padding: .55rem .8rem; text-align: left; border-bottom: 1px solid #f1ebee; color: #3b3b4f; }
      .table th { background: #f8f1f5; font-weight: 700; }
      .table tr.best td { background: #fdf3f8; font-weight: 600; }
      .footer { text-align: center; color: #9a9aaa; font-size: .85rem; margin-top: 2.5rem; }
      .footer a, .about a { color: #c8559c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌸 Flower Classifier</h1>
      <p>Upload a photo and a MobileNetV2 transfer-learning model tells you which flower it shows.</p>
      <div class="chips"><div class="chip">🌹 Rose</div><div class="chip">🌻 Sunflower</div>
      <div class="chip">🌷 Tulip</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_app, tab_about = st.tabs(["🔍 Classifier", "📖 About the project"])

# =====================================================================================================================
# Classifier
# =====================================================================================================================
with tab_app:
    ORDER = list(MODEL_INFO)  # the recommended model first, unknown models last
    model_files = sorted(
        MODELS_DIR.glob("*.keras"), key=lambda p: (ORDER.index(p.name) if p.name in ORDER else len(ORDER), p.name)
    )
    if not model_files:
        st.error("No model found in `models/`. Save a model there in step 5 of the notebook first.")
        st.stop()

    # --- Step 1: model -------------------------------------------------------------------------------------------------
    st.markdown('<div class="step"><span>1</span>Choose a model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">All three models share a MobileNetV2 network pretrained on ImageNet and differ in how their '
        "classification head was trained. If you are unsure, keep the recommended one.</div>",
        unsafe_allow_html=True,
    )
    model_path = st.selectbox(
        "Model",
        model_files,
        format_func=lambda p: MODEL_INFO.get(p.name, {}).get("name", p.stem),
        label_visibility="collapsed",
    )
    info = MODEL_INFO.get(model_path.name)
    if info:
        st.markdown(
            f'<div class="card model-card">{info["about"]}<br>'
            f'Validation accuracy <b>{info["val"]:.1%}</b> · Test accuracy <b>{info["test"]:.1%}</b></div>',
            unsafe_allow_html=True,
        )

    # --- Step 2: photo -------------------------------------------------------------------------------------------------
    st.markdown('<div class="step"><span>2</span>Add a photo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Upload a clear photo of a <b>rose</b>, a <b>sunflower</b> or a <b>tulip</b> (JPG or PNG). '
        "Close-ups of the flower head work best; wide shots of whole fields are harder. "
        "Any other picture is reported as <i>no match found</i>.</div>",
        unsafe_allow_html=True,
    )
    tab_upload, tab_example = st.tabs(["📤 Upload your photo", "🖼️ Try an example"])
    with tab_upload:
        upload = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with tab_example:
        st.caption("Flower photos from the test set, which the models never saw in training, and one that is not a flower.")
        example = st.pills("Example photo", list(EXAMPLES), label_visibility="collapsed")

    photo_bytes = upload.getvalue() if upload is not None else (EXAMPLES[example].read_bytes() if example else None)

    # --- Step 3: result ------------------------------------------------------------------------------------------------
    if photo_bytes is not None:
        st.markdown('<div class="step"><span>3</span>Result</div>', unsafe_allow_html=True)
        try:
            with st.spinner("Looking at the photo… (the first run also prepares the model, about 10 seconds)"):
                probabilities, similarity = classify(str(model_path), photo_bytes)
        except tf.errors.InvalidArgumentError:
            st.error("This file could not be read as an image. Please upload a JPG or PNG photo.")
            st.stop()

        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        flower = CLASSES[best]
        no_match = similarity < MIN_SIMILARITY or confidence < MIN_CONFIDENCE

        bars = "".join(
            f'<div class="bar-row"><div class="bar-name">{FLOWERS[c]["emoji"]} {c.capitalize()}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p:.1%};background:{FLOWERS[c]["color"]}">'
            f'</div></div><div class="bar-value">{p:.0%}</div></div>'
            for c, p in zip(CLASSES, probabilities)
        )

        if similarity < MIN_SIMILARITY:
            card = (
                '<div class="result nomatch"><div class="label">Result</div><div class="answer">🔍 No match found</div>'
                '<div class="sub">This photo does not look like a rose, a sunflower or a tulip, so the model does not '
                "give an answer. Try a clear photo of one of these three flowers.</div></div>"
            )
        elif confidence < MIN_CONFIDENCE:
            card = (
                '<div class="result nomatch"><div class="label">Result</div><div class="answer">🤔 No match found</div>'
                '<div class="sub">The model is not confident enough that this is a rose, a sunflower or a tulip. It '
                "may be another kind of flower, or the flower is too small or unclear in the photo.</div></div>"
            )
        else:
            card = (
                f'<div class="result"><div class="label">Prediction</div>'
                f'<div class="answer" style="color:{FLOWERS[flower]["color"]}">{FLOWERS[flower]["emoji"]} '
                f"{flower.capitalize()}</div>"
                f'<div class="sub">{confidence:.0%} confident</div>{bars}</div>'
            )

        col_photo, col_result = st.columns([1, 1.15], gap="large")
        with col_photo:
            st.image(photo_bytes, width="stretch")
        with col_result:
            st.markdown(card, unsafe_allow_html=True)
            if no_match:
                with st.expander("Show the model's raw probabilities anyway"):
                    st.markdown(bars, unsafe_allow_html=True)
                    st.caption(
                        f"Similarity to the training photos: {similarity:.2f} (a match needs at least {MIN_SIMILARITY})"
                    )

    st.caption("Want to know how the model was built and how well it works? Open the **📖 About the project** tab.")

# =====================================================================================================================
# About the project
# =====================================================================================================================
with tab_about:
    st.markdown(
        """
<div class="about">

<h3>🎯 The problem</h3>
<p>A garden centre lets customers upload a photo of a flower from their garden and sends back a care guide: when to
water, when to cut back, which pests to watch for. Today a member of staff looks at every photo and picks the guide by
hand. The goal: <b>choose the guide automatically</b> for the three most requested flowers, roses, sunflowers and
tulips, and find out <b>when the model can be trusted</b> and when a person should check its answer.</p>

<div class="stats">
  <div class="stat"><div class="value">390</div><div class="name">photos in total</div></div>
  <div class="stat"><div class="value">100</div><div class="name">training photos per flower</div></div>
  <div class="stat"><div class="value">93.3%</div><div class="name">test accuracy (28 of 30)</div></div>
  <div class="stat"><div class="value">2.3 M</div><div class="name">frozen pretrained weights reused</div></div>
</div>

<h3>🗂️ The data</h3>
<p>390 colour photos from the <a href="https://www.tensorflow.org/datasets/catalog/tf_flowers" target="_blank">TensorFlow
flower photos</a> collection (Flickr, CC BY 2.0), one folder per flower: <b>100 training</b>, <b>20 validation</b> and
<b>10 test</b> photos per class. The photos vary a lot: single close-ups, whole fields, bouquets in vases, people and
buildings in the background. The test photos stayed locked until the very end.</p>

<h3>🛠️ How it works</h3>
<div class="flow">
  <div class="card"><div class="num">STEP 1</div><b>Load</b><br>Photos read from folders and resized to
    160×160 pixels.</div>
  <div class="card"><div class="num">STEP 2</div><b>Reuse</b><br>MobileNetV2 pretrained on ImageNet, frozen, turns a
    photo into 1,280 features.</div>
  <div class="card"><div class="num">STEP 3</div><b>Train a head</b><br>Dropout + a 3-way softmax layer learns roses,
    sunflowers and tulips.</div>
  <div class="card"><div class="num">STEP 4</div><b>Augment</b><br>Random flips, rotations and zooms against
    overfitting.</div>
  <div class="card"><div class="num">STEP 5</div><b>Evaluate</b><br>Validation to choose, locked test set to report,
    error analysis.</div>
</div>
<p>Pixels are rescaled to [-1, 1] inside the model, so the app passes the raw photo and gets three probabilities
back. On top of that, the app checks whether a photo looks like the training photos at all, by comparing its
MobileNetV2 features with the 300 training photos, and answers <b>“no match found”</b> for anything else.</p>

<h3>📊 Results</h3>
<table class="table">
  <tr><th>Model</th><th>Training</th><th>Validation</th><th>Test</th></tr>
  <tr class="best"><td>Dense head ⭐</td><td>15 epochs, Adam 1e-3</td><td>86.7%</td><td>93.3%</td></tr>
  <tr><td>Augmented head</td><td>+ flips, rotations, zooms, 20 epochs</td><td>80.0%</td><td>96.7%</td></tr>
  <tr><td>Augmented head, extra training</td><td>+ 15 epochs, Adam 1e-5</td><td>80.0%</td><td>96.7%</td></tr>
</table>
<p style="font-size:.85rem;color:#8a8a9a;margin-top:.4rem">⭐ The final model was chosen by validation accuracy,
before the test set was opened. 60 validation and 30 test photos: one photo is worth 1.7 and 3.3 percentage
points.</p>

<h3>💡 Key findings</h3>
<ul>
  <li><b>Transfer learning works with very little data.</b> With only 300 training photos, a frozen MobileNetV2 and a
      single dense layer classify 28 of 30 unseen test photos correctly.</li>
  <li><b>Choose by validation, report the test.</b> The augmented models score one photo higher on the test set
      (96.7%), but choosing them for that would make the test score optimistic. With 30 test photos, one photo is
      random variation.</li>
  <li><b>The dense head overfits</b>: about 99–100% training accuracy against 87% validation accuracy. Data
      augmentation did not raise the validation accuracy here (80.0%).</li>
  <li><b>Both test mistakes are sunflowers</b> (below): one in a wide garden shot, one in a mixed display with other
      flowers. Close-ups of one kind of flower work best.</li>
  <li><b>Mistakes come with lower confidence.</b> On the validation and test photos, answers below 80% confidence
      include 6 of the 10 mistakes but only 7 of the 80 correct answers, so they are worth a staff check.</li>
  <li><b>A 3-class model always answers.</b> It called a photo of a forest by the sea a sunflower with 95% confidence,
      so the app adds a similarity check. Look-alike flowers such as peonies or daisies can still pass it.</li>
</ul>
<h3>🔎 The two misclassified test photos</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    MISTAKES = [
        ("2807106374_f422b5f00c.jpg", "🌹 rose (72%)", "A wide shot of a garden: the sunflowers are small and far away."),
        ("2733109082_1351f6738a_n.jpg", "🌷 tulip (79%)", "Sunflowers in vases next to other flowers, some dark red."),
    ]
    for col, (name, predicted, why) in zip(st.columns(2, gap="medium"), MISTAKES):
        with col:
            st.image(str(TEST_DIR / "sunflower" / name), width="stretch")
            st.markdown(
                f'<div class="card">True: 🌻 sunflower · Predicted: {predicted}<br>{why}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
<div class="about">
<h3>🚀 Next steps</h3>
<ul>
  <li>Fine-tune the top layers of MobileNetV2 with a low learning rate (the base stayed frozen in this project).</li>
  <li>Collect more photos, especially wide shots, and evaluate with cross-validation for more reliable numbers.</li>
  <li>Add more flowers and an explicit <i>other</i> class to handle out-of-scope photos better.</li>
</ul>

<h3>🧰 Tech stack</h3>
<div class="chips left">
  <div class="chip">Python 3.13</div><div class="chip">TensorFlow / Keras 3</div><div class="chip">MobileNetV2</div>
  <div class="chip">scikit-learn</div><div class="chip">pandas</div><div class="chip">Matplotlib</div>
  <div class="chip">Streamlit</div><div class="chip">uv</div><div class="chip">Streamlit Community Cloud</div>
</div>

<p style="margin-top:1rem">The full analysis is in the notebook on
<a href="{REPO_URL}" target="_blank">GitHub</a>.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Built with TensorFlow / Keras and Streamlit · Photos: TensorFlow flower photos (CC BY 2.0) · '
    f'<a href="{REPO_URL}" target="_blank">Source on GitHub</a></div>',
    unsafe_allow_html=True,
)
