"""Detection server: POST /predict (menu detection via OpenRouter) and POST /leftover (% of plate left).

/leftover: multipart/form-data `image` (before meal), `leftover_image` (after meal),
`metadata` (optional JSON from /predict; validated but not yet used).
"""

import base64
import json
import os
import uuid
from io import BytesIO

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image

from pipeline import LeftoverPipeline, UnusableImage

API_KEY = os.environ.get("OPENROUTER_API_KEY")

DATA_PATH = "data.json"
STATIC_DIR = "static"

def build_detection_prompt(food_names):
    menu_list = "\n".join(f"- {name}" for name in food_names)
    return f"""\
Detect every distinct food dish in this image. Respond with ONLY a JSON array
(no markdown, no extra text), one object per dish, in this exact shape:

[
  {{
    "label": "the matching menu name, copied verbatim",
    "confidence": 0.0 to 1.0,
    "box_2d": [ymin, xmin, ymax, xmax]
  }}
]

For "label", you MUST choose the closest matching name from this exact menu list
and copy it verbatim (do not reword, translate, or invent a new label):
{menu_list}

If a dish in the image does not clearly match any name on that list, leave it
out of the results entirely.

box_2d values must be normalized to a 0-1000 scale relative to the image
width/height (not raw pixels).
"""


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_menu_data(path):
    with open(path, "r", encoding="utf-8") as f:
        menus = json.load(f)
    return {menu["name_en"]: menu["id"] for menu in menus}


def parse_detections(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    return json.loads(text)


def to_pixel_bbox(box_2d, image_width, image_height):
    ymin, xmin, ymax, xmax = box_2d
    return {
        "x1": round(xmin / 1000 * image_width, 2),
        "y1": round(ymin / 1000 * image_height, 2),
        "x2": round(xmax / 1000 * image_width, 2),
        "y2": round(ymax / 1000 * image_height, 2),
    }


def classify(image_path, data_path=DATA_PATH):
    """Run detection on image_path. Returns (image, predictions).

    Raises RuntimeError if the OpenRouter call fails or its response can't be parsed.
    """
    image = Image.open(image_path)
    image_width, image_height = image.size
    base64_image = encode_image(image_path)
    menu_lookup = load_menu_data(data_path)
    menu_lookup_ci = {name.lower(): menu_id for name, menu_id in menu_lookup.items()}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Food Detector",
    }

    payload = {
        "model": "google/gemini-2.5-flash-image",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_detection_prompt(menu_lookup.keys())},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    result = response.json()

    if response.status_code != 200 or "choices" not in result:
        raise RuntimeError(f"OpenRouter API error: {result}")

    content = result["choices"][0]["message"]["content"]

    try:
        detections = parse_detections(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"Could not parse model response: {content}")

    predictions = []
    for item in detections:
        label = item["label"]
        menu_id = menu_lookup_ci.get(label.strip().lower())
        if menu_id is None:
            continue
        predictions.append(
            {
                "bbox": to_pixel_bbox(item["box_2d"], image_width, image_height),
                "confidence": item["confidence"],
                "label": label,
                "menu_id": menu_id,
            }
        )

    return image, predictions


app = Flask(__name__)
os.makedirs(STATIC_DIR, exist_ok=True)
pipeline = LeftoverPipeline()  # load models once, not per request


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file path"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        unique_id = uuid.uuid4()
        img_path = f"{STATIC_DIR}/{unique_id}.jpg"
        file.save(img_path)

        processed_img, predictions = classify(img_path)

        if processed_img.mode == 'RGBA':
            processed_img = processed_img.convert('RGB')

        img_byte_arr = BytesIO()
        processed_img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        return jsonify({
            'message': 'Prediction completed',
            'metadata': predictions,
            'image': img_base64
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


def read_image(name: str) -> np.ndarray:
    upload = request.files.get(name)
    if upload is None:
        raise KeyError(f"missing file field '{name}'")
    image = cv2.imdecode(np.frombuffer(upload.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UnusableImage(f"'{name}' is not a readable image")
    return image


@app.post("/leftover")
def leftover():
    try:
        raw_metadata = request.form.get("metadata")
        if raw_metadata:
            json.loads(raw_metadata)
        before_img = read_image("image")
        after_img = read_image("leftover_image")
        result = pipeline.compare(before_img, after_img)
    except (KeyError, json.JSONDecodeError) as e:
        return jsonify(error=str(e.args[0] if isinstance(e, KeyError) else f"metadata is not valid JSON: {e}")), 400
    except UnusableImage as e:
        return jsonify(error=str(e)), 422
    return jsonify(result)


if __name__ == "__main__":
    # debug stays off by default: the reloader would load the models twice and the debugger is unsafe on 0.0.0.0
    app.run(host="0.0.0.0", port=6000, debug=os.environ.get("FLASK_DEBUG") == "1")
