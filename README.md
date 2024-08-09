![github-banner](https://github.com/user-attachments/assets/b5193604-1138-4aa5-8ef4-fb59e0bcba57)
#### Brittany Chan, Joel Baltodano, Sakshi Nikte, and Audrey Reinhard 
#### In collaboration with Halle Dimsdale-Zucker for UCR Data Science Pathway Fellowship 2024 
## About This Project
Halle Dimsdale-Zucker of the UCR Psychology Dept. and her lab,  Dimsdale-Zucker Memory and Context Lab (DZMaC), are performing ongoing research to study the recall of complex content in the form of TV shows and commercials. Her goal is to analyze shared neural signals of multiple participants who are exposed to the same stimuli and accomplishes this by incorporating functional magnetic resonance imaging ([fMRI](https://en.wikipedia.org/wiki/Functional_magnetic_resonance_imaging)). Participants are exposed to multiple stimuli then asked, while inside an fMRI machine, to freely recall as much of the events as possible. After this audio is recorded, they are timestamped such that we can relate the segments of the audio to the imaging of the fMRI scans, then transcribed. After finishing the timestamps, her and her team began the transcription process and the matching of the content to events in the given TV shows and commercials; this task proved to be monotonous and time-consuming.

Our team was tasked with automating the process of classifying the recall transcriptions to the content. We went about this task by incorporating an AI transcription tool, [Revoldiv](https://revoldiv.com), and an advanced NLP transformer, [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert), for the basis of our model. After transcribing the rest of the audio, preprocessing, and training, we have a final result. Our model is not as perfect as needed due to a noticeable limitation in available data, so the task still requires human intervention. We imagine that given more data, DistilBERT would prove to be a completely independent tool for this task.

## Required Tools
#### Some Used Dependencies/Modules:
  * flask, flask_cors
  * nltk, nltk.tokenize
  * json
  * os
  * torch

### Necessary Downloads:
If you don't have it already, install Python [here](https://www.python.org/downloads/).

Using **pip**:
* `pip install transformers pandas numpy torch`

Using **conda**:
* `conda install -c conda-forge transformers pandas numpy`
* `conda install pytorch -c pytorch`

## Input Formatting
Transcriptions made by Revoldiv are saved in a very unique format, as such the model accepts transcriptions in the following format:
<br><img width="343" alt="Screenshot 2024-08-06 at 9 53 01 AM" src="https://github.com/user-attachments/assets/c02d4e4d-10d4-4c21-9ec8-de328ac0b4a1">
The program also handles transcriptions by Dr. Halle's team, which are in the following format:
<br><img width="421" alt="Screenshot 2024-08-06 at 10 51 39 PM" src="https://github.com/user-attachments/assets/346aa995-f429-40df-80e6-09904989a70f"> 
## Using the Fine-tuned Model with Front-end
* Run `predict.py` to handle back-end processes.
  * Python backend is ready! <br><img width="490" alt="Screenshot 2024-08-06 at 10 12 04 AM" src="https://github.com/user-attachments/assets/73954842-3942-4a77-854f-140f66f2af31">

* On VSCode, click <img width="79" alt="Screenshot 2024-08-06 at 9 34 47 AM" src="https://github.com/user-attachments/assets/598401e4-b4f7-4244-af7a-88013a8f483a"> to initialize the front-end.
  * Note: The ports used are `5500` and `5000`
* Choose the file to transcribe then press "**Upload**".
* Now, you will find your transcribed file in the folder "**classified-transcripts**"
<br><img width="185" alt="Screenshot 2024-08-06 at 11 06 13 AM" src="https://github.com/user-attachments/assets/6d70cf5e-0236-41f9-a4fe-b1a62939ea9a">
## Webpage Preview
<img width="1233" alt="Screenshot 2024-08-06 at 9 41 00 AM" src="https://github.com/user-attachments/assets/0b93f02b-cc61-4c60-9367-2b483a1eb2b9">
