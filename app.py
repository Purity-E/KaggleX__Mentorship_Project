from flask import Flask, request, render_template
import os
from transformers import ViTImageProcessor, ViTForImageClassification, LlamaForCausalLM, LlamaTokenizer
from PIL import ImageDraw, ImageFont, Image
import torch
from tqdm import tqdm
from datasets import load_metric
import shutil


app = Flask(__name__)

# Loading the vit fine-tuned model and tokenizer
vit_model = ViTForImageClassification.from_pretrained('vit_bird_species')
vit_processor = ViTImageProcessor.from_pretrained('vit_bird_species')

# loading ocra-mini-3b model and tokenizer
ocra_tokenizer = LlamaTokenizer.from_pretrained("ocra_mini_3b")
ocra_model = LlamaForCausalLM.from_pretrained('ocra_mini_3b')

# Defining a function to process the uploaded image and make predictions
def predict_image(file_path):
    img = Image.open(file_path)

    # processing the test image
    inputs = vit_processor(images=img, return_tensors="pt")

    # Making predictions while temporarily disabling gradient computation
    vit_model.eval()  # Setting the model to evaluation mode (e.g., for dropout layers)
    with torch.no_grad():  # Enter the no_grad context
        predictions = vit_model(**inputs)  # Perform inference

    # getting the logits
    logits = predictions.logits

    # getting the class index with the highest probability
    class_indices = int(torch.argmax(logits, dim=1))
    # printing the label
    species_class = vit_model.config.id2label[class_indices]
    
    # making corrections for the TOUCAN label
    if species_class == 'TOUCHAN':
        species_class = 'TOUCAN'
    else:
        species_class = species_class

    #getting predictions from ocra-mini model
    prompt = f"Write about {species_class},its habitat, diet, behavior, and conservation status"

    tokens = ocra_tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.2, 'generate_len': 200, 'top_k': 30}

    # generating the model
    length = len(tokens[0])
    with torch.no_grad():
        pred_result = ocra_model.generate(
                    input_ids=tokens,
                    max_length=length+instance['generate_len'],
                    top_p=instance['top_p'],
                    temperature=instance['temperature'],
                    top_k=instance['top_k'],
                    use_cache=True,
                    do_sample=True,
            )
        
    output = pred_result[0][length:]
    
    decoded_predictions = ocra_tokenizer.decode(output, skip_special_tokens=True)
    
    return decoded_predictions, species_class

# Define a route to handle the home page

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    uploaded_image = None
    species = None

    if request.method == 'POST' and 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)
            prediction_result,species = predict_image(image_path)
            uploaded_image = image_file.filename

            # moving the image to static folder
            static_path = os.path.join('static', image_file.filename)
            
            # Move the file from the source to the destination
            shutil.move(image_path, static_path)

    return render_template('home.html', prediction_result=prediction_result, species=species, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)