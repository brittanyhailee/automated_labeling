![github-banner](https://github.com/user-attachments/assets/b5193604-1138-4aa5-8ef4-fb59e0bcba57)
#### Brittany Chan, Joel Baltodano, Sakshi Nikte, and Audrey Reinhard 
#### In collaboration with Halle Dimsdale-Zucker for UCR Data Science Pathway Fellowship 2024 
## About this project
* This project aims to observe neural signals in the brain by asking participants in the experiment to watch 3 ten minute video clips composed of 48 segments. The participants do their recall process in the fMRI machine where their audio is recorded to record what segment they recall and align them to the fMRI images. Our task is to automate the process of classifying recall transcriptions and propose a tool that future researchers may build on to help in automating categorization in their experiments.
## Set-Up
* Used Dependencies/Modules:
  * **flask, flask_cors**
  * **nltk, nltk.tokenize**
  * **json**
  * **os**
  * **torch**
* Transcriptions were made with Revoldiv AI thus, functions handle a specific format.
  * Example transcription:
    <br><img width="343" alt="Screenshot 2024-08-06 at 9 53 01 AM" src="https://github.com/user-attachments/assets/c02d4e4d-10d4-4c21-9ec8-de328ac0b4a1">

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
