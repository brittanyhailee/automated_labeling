from flask import Flask, request, jsonify
from flask_cors import CORS 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from formatting import read_recall_new
import json

import torch

import numpy as np
print(np.__version__)

app = Flask(__name__)
# CORS(app, resources={r"/*": {'Access-Control-Allow-Origin': "*"}})
CORS(app)

model_name = 'brittanyhlc/automated-labeling-distilbert'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
categories = ['fillmore_ep-7_236-406_part1_resized_1280-720',
 'area-wide-exterminators_resized_1280-720',
 'fillmore_ep-7_236-406_part2_resized_1280-720',
 'arctic-leash_resized_1280-720',
 'the-tribe_episode-1_1202-1332_part-1_resized_1280-720',
 'arrowhead-clinic_resized_1280-720',
 'the-tribe_episode-1_1202-1332_part-2_resized_1280-720',
 'bang-bang-boom-boom_resized_1280-720',
 'fillmore_ep-14_828-958_part1_resized_1280-720',
 'big-al-pizza_resized_1280-720',
 'fillmore_ep-14_828-958_part2_resized_1280-720',
 'bowling_resized_1280-720',
 'the-tribe_episode-6_645-815_part1_resized_1280-720',
 'cecil-myers_resized_1280-720',
 'the-tribe_episode-6_645-815_part2_resized_1280-720',
 'china-star-delicious_resized_1280-720',
 'corner-gas_s1-ep1_636-805_part1_resized_1280-720',
 'crazy-gideon_resized_1280-720',
 'corner-gas_s1-ep1_636-805_part2_resized_1280-720',
 'go-wireless_resized_1280-720',
 'the-riches_s1-ep1_2529-2659_part1_resized_1280-720',
 'heindl-used-cars_resized_1280-720',
 'the-riches_s1-ep1_2529-2659_part2_resized_1280-720',
 'horrible-insurance_resized_1280-720',
 'corner-gas_s2-ep1_805-935_part1_resized_1280-720',
 'jp-audio-mart_resized_1280-720',
 'corner-gas_s2-ep1_805-935_part2_resized_1280-720',
 'lens-rap_resized_1280-720',
 'the-riches_s2-ep1_3244-3415_part1_resized_1280-720',
 'local-commercial-fail_resized_1280-720',
 'the-riches_s2-ep1_3244-3415_part2_resized_1280-720',
 'martin-fine-furniture_resized_1280-720',
 'panam_s01ep01_2310-2349_resized_1280-720',
 'village-inn-family-buffet_resized_1280-720',
 'panam_s01ep01_258-347_resized_1280-720', 'take-my-wife_resized_1280-720',
 'popular_ep-04_1332-1502_part1_resized_1280-720',
 'pay-me-now_resized_1280-720',
 'popular_ep-04_1332-1502_part2_resized_1280-720',
 'rug-it-riders_resized_1280-720', 'panam_s01ep02_316-359_resized_1280-720',
 'cie-perfume_resized_1280-720', 'panam_s01ep02_746-831_resized_1280-720',
 'sushi-bar', 'popular_ep-05_2131-2308_part1_resized_1280-720',
 'adopt-a-pet_resized_1280-720',
 'popular_ep-05_2131-2308_part2_resized_1280-720',
 'henry-beer-commercial_resized_1280-720']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    content = file.read().decode('utf-8')

    sentences = read_recall_new(content)
    print('sentences:', sentences)
    content_to_string = json.dumps(content)
    # print(content_to_string) 
    # print(content_to_string)
    return jsonify({'prediction': content})
   

    text = 'hi this is fun'
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1).item()

    return jsonify({'prediction': categories[predictions]})

    # return jsonify({'prediction': f'File {filename} received'})

    # with open(filename, 'r') as file:
    #     file.readlines()
    #     # print(file)
    
    # return jsonify({'prediction': file})

    

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    
