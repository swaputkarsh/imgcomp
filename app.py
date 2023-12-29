from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import matplotlib.image as mpimg
import torch
from sentence_transformers import util
import open_clip
from io import BytesIO

# Now run the Flask App
app = Flask(__name__)
CORS(app)
# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)


def image_encoder(img_data):
    try:
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        img_encoded = model.encode_image(img_tensor)
        return img_encoded
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def generate_score(img1_data, img2_data):
    img1 = image_encoder(img1_data)
    img2 = image_encoder(img2_data)

    if img1 is None or img2 is None:
        return {"error": "Error processing images. Please check the file data."}

    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    if score < 70:
        score = 0
    result = {
        "similarity_score": score,
        "comparison_result": 'images are similar' if score >= 70 else 'images are different'
    }
    return result


@app.route("/compare", methods=["POST"])
def compare_images():
    try:
        img1_data = request.files["image1"].read()
        img2_data = request.files["image2"].read()

        result = generate_score(img1_data, img2_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})


@app.route("/show", methods=["POST"])
def show_images():
    try:
        img1_data = request.files["image1"].read()
        img2_data = request.files["image2"].read()

        image1 = mpimg.imread(BytesIO(img1_data))
        image2 = mpimg.imread(BytesIO(img2_data))

        result = generate_score(img1_data, img2_data)

        return jsonify({
            "comparison_result": result["comparison_result"],
            "similarity_score": result["similarity_score"],
            "image1": image1.tolist(),
            "image2": image2.tolist()
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})


if __name__ == "__main__":
    app.run(debug=True)
