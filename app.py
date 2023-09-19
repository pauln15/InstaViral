from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image

from image_predictor import ImagePredictor

app = Flask(__name__, static_folder='build', static_url_path='/')

CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'model-resnet50.pth'
predictor = ImagePredictor(model_path, device)


@app.route("/", defaults={'path': ''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/test')
def index():
    return jsonify({'message': 'Flask backend is online!'})


@app.route('/api', methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    files = request.files.getlist('file')

    if not files or '' in (file.filename for file in files):
        return jsonify({'message': 'No selected file'}), 400

    scores = {}
    for file in files:
        # Check if the file is an image
        if not (file.content_type.startswith('image/') or file.filename.endswith('.HEIC')):
            return jsonify({'message': f'{file.filename} is not an image'}), 400

        # Open the image file
        image = Image.open(file.stream)
        # Pass the image to the predictor
        score = predictor.predict(image)
        # Save the score for this file
        scores[file.filename] = score

    return jsonify({'Popularity scores': scores})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
