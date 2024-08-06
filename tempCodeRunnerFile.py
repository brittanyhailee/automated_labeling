ons_path):
    file_path = os.path.join(new_transcriptions_path, filename)
    paragraph = transcript_to_paragraph(file_path)
    paragraph = sent_tokenize(paragraph)

    for lines in paragraph: 
        classified = classify(lines)
        output_filename = filename.replace('transcript.txt', 'classified.txt')
        with open(output_filename, 'a') as outfile:
            outfile.write(lines + '\n' + classified + '\n' )
