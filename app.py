from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import matplotlib.image as mpimg
import torch
from sentence_transformers import util
import open_clip
from io import BytesIO
import datetime
import os

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

def rename_and_save_images(img1_data, img2_data, output_folder):
    try:
        img1 = Image.open(BytesIO(img1_data)).convert('RGB')
        img2 = Image.open(BytesIO(img2_data)).convert('RGB')

        current_datetime = datetime.datetime.now()
        formatted_datetime_for_filename = current_datetime.strftime("%Y%m%d%H%M%S").replace(":", "_")

        new_image1_name = f"{formatted_datetime_for_filename}_org_img.jpg"
        new_image2_name = f"{formatted_datetime_for_filename}_com_img.jpg"

        img1_output_path = os.path.join(output_folder, new_image1_name)
        img2_output_path = os.path.join(output_folder, new_image2_name)

        img1.save(img1_output_path)
        img2.save(img2_output_path)

        return {"success": "Images saved and renamed successfully."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.route("/compare", methods=["POST"])
def compare_images():
    try:
        img1_data = request.files["image1"].read()
        img2_data = request.files["image2"].read()

        # Save and rename images
        output_folder = r"/home/imgcomp/data"  # Change this to your desired output folder
        save_result = rename_and_save_images(img1_data, img2_data, output_folder)

        # Calculate similarity score
        result = generate_score(img1_data, img2_data)
        result.update(save_result)

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
        result.update({"image1": image1.tolist(), "image2": image2.tolist()})

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})

if __name__ == "__main__":
    app.run(debug=True)
