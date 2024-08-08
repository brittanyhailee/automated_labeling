from flask import Flask, request, jsonify, send_file
from flask_cors import CORS 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from formatting import read_recall_new, transcript_to_paragraph
import json
import os
import torch
import numpy as np
import pandas as pd 

print(np.__version__)

app = Flask(__name__)
# CORS(app, resources={r"/*": {'Access-Control-Allow-Origin': "*"}})
CORS(app)

classified_file_path = '/Users/brittany/Desktop/DS_Fellowship/automated_labeling/classified-transcripts'
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=228)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform inference
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    class_labels = [label for label in categories]

    predicted_class = class_labels[predictions.item()]
    # print(f"\t-predicted class: {predicted_class}\n")
    # return f"\t-predicted class: {predicted_class}\n"
    return predicted_class



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    input_filename = file.filename
    if not os.path.isdir(classified_file_path):
            os.makedirs(classified_file_path)

    if input_filename.endswith('.txt'): 
        file_content = file.read().decode('utf-8')
        transcriptions = read_recall_new(file_content)
        file.seek(0)  # Reset  file pointer to beginning after reading for processing
        paragraph = transcript_to_paragraph(file)
        paragraph = sent_tokenize(paragraph)
    
        output_filename = input_filename.replace('transcript.txt', 'classified.txt')
        output_filename = os.path.join(classified_file_path, output_filename)

        for lines in paragraph: 
            classified = "\t-predicted class: "

            classified += classify(lines)
            with open(output_filename, 'a') as outfile:
                outfile.write(lines + '\n' + classified + '\n\n' )

    elif input_filename.endswith('.csv'): 
        input_file_path = os.path.join(classified_file_path, input_filename)  # Combine with classified_file_path
        file.save(input_file_path) 
        transcript_df = pd.read_csv(input_file_path)
        labels = ['run_ID', 'segment_name']
        transcript_df.drop(labels, axis=1,inplace=True)
        transcript_df.dropna(subset=['text'], inplace=True)
        output_filename = input_filename.replace('.csv', '(classified).csv')
        output_filename = os.path.join(classified_file_path, output_filename)
        new_df = pd.DataFrame(columns=['starttime', 'endtime','segment_id','text'])
        rows = []

        for index, row in transcript_df.iterrows():
            sentence = row['text']
            rows.append({'starttime': str(row['starttime']), 'endtime': str(row['endtime']),'segment_id': classify(sentence), 'text': row['text']})
            # classifications.append(classify(sentence))

        new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)
        new_df.to_csv(output_filename, index=False)
        os.remove(input_file_path)

        
    print('output filename is ' + output_filename)
    return jsonify({'categorized': output_filename})
   

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    
