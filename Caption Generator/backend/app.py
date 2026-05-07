from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)

# Load BLIP model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

@app.route('/generate-caption', methods=['POST'])
def generate_caption():

    image_file = request.files['image']

    image = Image.open(image_file).convert('RGB')

    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({
        "caption": caption
    })

if __name__ == '__main__':
    app.run(debug=False)