from flask import Flask, request, jsonify, send_file
from flask_cors import CORS 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from predict import classify
import json
import os
import torch
import numpy as np
import chardet
import pandas as pd

def detect_encoding(file_path):
    with open(file_path, 'rb') as file: 
        detector = chardet.universaldetector.UniversalDetector() 
        for line in file:
            detector.feed(line)
            if detector.done:
                break

    detector.close()
    return detector.result['encoding']


def transcript_to_paragraph(file_path, encoding):
    text = []
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
            
        if line and not line.startswith("00:"): # remove timestamps
            text.append(line)
            
    paragraph = " ".join(text)
    paragraph.replace('\n', ' ')
    
    return paragraph



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

print("Input the format of transcript file")
fileformat = input("\t('c': csv, 't':txt) ")

if fileformat == 't': 
    new_transcriptions_path = input("\nEnter folder path containing transcriptions by Revoldiv: ")

    while os.path.isdir(new_transcriptions_path) is False:
        print(f"\nThe path `{new_transcriptions_path}` does not exist.")
        new_transcriptions_path = input("Enter folder path containing transcriptions by Revoldiv: ")


    classified_transcriptions_path = input("\nEnter path to store the classified transcripts: ")
    while os.path.isdir(classified_transcriptions_path) is False:
        print(f"\nThe path `{classified_transcriptions_path}` does not exist.")
        classified_transcriptions_path = input("Enter path to store the classified transcripts: ")

    for filename in os.listdir(new_transcriptions_path):
        file_path = os.path.join(new_transcriptions_path, filename)
        encoding = detect_encoding(file_path)
        paragraph = transcript_to_paragraph(file_path, encoding)
        paragraph = sent_tokenize(paragraph)
        output_filename = os.path.join(classified_transcriptions_path, filename.replace('transcript.txt', 'classified.txt'))

        for lines in paragraph: 
            classified = classify(lines)
            with open(output_filename, 'a') as outfile:
                outfile.write(lines + '\n' + '\t-'+ classified + '\n' )


elif fileformat == 'c':
    csv_transcriptions_path = input("Enter folder path containing transcriptions in .csv: ")

    while os.path.isdir(csv_transcriptions_path) is False:
        print(f"\nThe path `{csv_transcriptions_path}` does not exist.")
        csv_transcriptions_path = input("Enter folder path containing transcriptions in .csv: ")

    classified_transcriptions_path = input("\nEnter path to store the classified transcripts: ")
    while os.path.isdir(classified_transcriptions_path) is False:
        print(f"\nThe path `{classified_transcriptions_path}` does not exist.")
        classified_transcriptions_path = input("Enter path to store classified transcripts: ")

    for filename in os.listdir(csv_transcriptions_path):
        file_path = os.path.join(csv_transcriptions_path, filename)
        transcript_df = pd.read_csv(file_path)
        labels = ['run_ID', 'segment_name']
        transcript_df.drop(labels, axis=1,inplace=True)
        transcript_df.dropna(subset=['text'], inplace=True)

        output_filename = filename.replace('.csv', '(classified).csv')
        output_filename = os.path.join(classified_transcriptions_path, output_filename)
  
        new_df = pd.DataFrame(columns=['starttime', 'endtime','segment_id','text'])
        rows = []

        for index, row in transcript_df.iterrows():
            sentence = row['text']
            rows.append({'starttime': str(row['starttime']), 'endtime': str(row['endtime']),'segment_id': classify(sentence), 'text': row['text']})
        new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)
        new_df.to_csv(output_filename, index=False)





        