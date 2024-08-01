def read_recall_new(content): # Differnet method for our very own hardworking transcribers!
    
    transcriptions = []
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     lines = file.read_lines()
    # # lines = content
    # print(lines)
    lines = content

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