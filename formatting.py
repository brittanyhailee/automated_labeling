def read_recall_new(file_content): # Differnet method for our very own hardworking transcribers!
    
    transcriptions = []
    lines = file_content.splitlines()

    # lines = content
    # print(lines)
    # lines = content

    timestamps, text = [], []
    curr_start_time = None # using this logic as we have multiline potential
    curr_end_time = None   # and dont know how long the current segment lasts
    curr_text = []

    for line in lines:
        line = line.strip()
        if not line:       # if empty line, skip
            continue
        if line.startswith("00:"): # if at a timestamp
            if curr_start_time and curr_end_time and curr_text:
                transcriptions.append((curr_start_time, curr_end_time, " ".join(curr_text)))
                curr_text = []
            times = line.split(" --> ")
            if len(times) == 2: # a sanity check for the most part
                curr_start_time = times[0].replace("00:", "", 1) # gets rid of the hour marker
                curr_end_time = times[1].replace("00:", "", 1)
        else:               # if text line, add to our curr segment content
            curr_text.append(line)

    if curr_start_time and curr_end_time and curr_text: # to ensure we dont miss last block (sanity check)
        transcriptions.append((curr_start_time, curr_end_time, " ".join(curr_text)))

    # print('the lines:', transcriptions)
    return transcriptions


def transcript_to_paragraph(file): 
    text = []
    
    # Read the content from the file-like object
    file_content = file.read().decode('utf-8')
    print("Raw file content:", file_content)
    
    # Split the content into lines
    lines = file_content.splitlines()
    
    for line in lines:
        line = line.strip()
        
        if line and not line.startswith("00:"):  # remove timestamps
            text.append(line)
    
    # Join text and replace new lines with spaces
    paragraph = " ".join(text)
    paragraph = paragraph.replace('\n', ' ')
    print(paragraph)
    return paragraph


# def transcript_to_paragraph(file_content):
#     text = []
#     paragraph = []

#     # lines = file_content.splitlines()
#     for line in file_content:
#         line = line.strip()

#         if line and not line.startswith("00:"):
#             text.append(line)
#             # parts = [part.strip() for part in line.split('.') if part.strip()]
#             # paragraph.extend(text)

#     # paragraph = paragraph.replace('\n', ' ')
#     paragraph = " ".join(text)
    
#     return paragraph