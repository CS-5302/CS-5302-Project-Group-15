# Speech Recognition and Translation System For Medical Communication

This is our final project for the course *CS 5302 - Speech and Language Processing With Generative AI* offered in Spring 2024 at the Lahore University of Management Sciences. This project was supervised by the course instructor Dr. Agha Ali Raza (Associate Professor at the Department of Computer Science, LUMS).

## Introduction

Effective communication between patients and healthcare providers is vital for quality care, especially considering the diverse linguistic backgrounds of patients. This diversity poses challenges, potentially leading to adverse outcomes for refugees, immigrants, and those in underserved areas. Our proposed solution is an SMTS (Speech-to-Machine Translation System), integrating speech recognition, machine translation, and text-to-speech technologies to bridge language gaps in healthcare. Development involves database selection, LLM fine-tuning, and RAG pipeline utilization, enabling real-time access to medical advice and information.

## System Design

Our overall SMTS system can be broken down into the following blocks:

- **Audio Input**: The system receives a user's query in the form of an audio input.
- **Speech-to-Text (STT) Module**: The audio query is transcribed into English using [FasterWhisper](https://github.com/SYSTRAN/faster-whisper).
- **Large Language Model (LLM) Utilization**: A pre-trained LLM is employed to interpret the transcribed query. The model has been fine-tuned on the [MeDAL](https://github.com/mcGill-NLP/medal) dataset to ensure it can provide medically accurate diagnoses in response to patient symptoms. For this purpose, we used [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/). The LLM generates a text response in English based on the user's query. This response contains the expert diagnosis for the patient.
- **Machine Translation**: The English text response is translated into the user's preferred language by employing [MarianMT](https://huggingface.co/docs/transformers/en/model_doc/marian).
- **Text-to-Speech (TTS) Module**: The translated text is converted back into audio form with [Google Translate's text-to-speech API](https://pypi.org/project/gTTS/).
- **Audio Output**: An audio recording of the diagnosis in the patient's language is played back.

![alt text](system_design_diagram.png)

