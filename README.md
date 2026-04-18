# AI Voice Fraud Detection System

**A deep learning based system designed to detect AI-generated speech in real-world audio, with applications in fraud prevention and telecommunication security.**


## Overview

Recent advances in Generative AI have made it possible to synthesise extremely realistic human voices using tools such as ElevenLabs, Hume AI, and Amazon Polly. While these technologies enable useful applications, they also introduce new risks:

* impersonation of bank officials, government authorities or family members
* voice-based phishing and scam calls
* identity spoofing using synthetic speech

This system is framed as a decision-support tool for detecting AI-generated scam calls, by developing a binary classification system to distinguish between REAL human speech and SYNTHETIC AI-generated audio.

The goal is to:
* flag suspicious audio
* assist users or institutions in identifying potential fraud

This project is not limited to model development, it explicitly evaluates the societal implications of deploying AI detection systems. The system is implemented as an ensemble of two deep learning models and deployed as a simple [Streamlit web-app](https://ai-audio-detector.streamlit.app/) for real-time inference.


## Technical Architecture

### Dataset

<u>Training dataset</u>: [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)

<u>Summary</u>:
* 1,866 samples
    * 933 real
    * 933 synthetic
* FLAC, 16 kHz mono-channel
* 2.5 to 13 seconds

### Model 1 - CNN trained on Mel Spectrograms

<u>Input</u>: log-Mel spectrograms (64 Mel bands)

<u>Architecture</u>:
* 3 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
* Adaptive pooling
* Fully connected classifier

<u>Captures</u>:
* spectral artifacts
* frequency-domain inconsistencies

Forkers can retrain the CNN from scratch to suit their own needs, using the `train_cnn.py` script.

### Model 2 — Fine-tuned wav2vec2

<u>Backbone</u>: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)

<u>Fine-tuning</u>: frozen encoder except last 2 transformer layers

<u>Input</u>: raw waveform (16 kHz mono-channel)

<u>Captures</u>:
* temporal speech patterns
* prosody and rhythm
* linguistic inconsistencies

Forkers can finetune wav2vec2 again to suit their own needs, using the `finetune_wav2vec2.py` script.

### Ensemble Strategy

The final prediction is computed as:

```
final_score = (cnn_score + wav2vec2_score) / 2
```

which combines:
* spectral learning (CNN)
* temporal + semantic learning (wav2vec2)

Forkers can evaluate the ensemble model on their own data, using the `infer.py` script.

### Web Application

The Streamlit web interface allows users to:
* upload audio files (.mp3, .wav, .opus, .flac)
* receive:
    * classification (REAL or SYNTHETIC)
    * confidence score
    * risk interpretation


## Evaluation

Real-world evaluation of the system was performed on a custom test set comprising:
* 26 real audio samples (YouTube videos, WhatsApp voice notes)
* 28 AI-generated samples (custom ElevenLabs voices)

where the REAL audio files are considered negative samples, while the SYNTHETIC files are considered positive. The evaluation results can be found in the `test_results.csv` file.

### Observations

* <u>Accuracy</u>: 92.59%
* <u>F1 Score</u>: 93.10%
* <u>Confusion Matrix</u>:
```
TP: 27    FN: 1
FP: 3     TN: 23
```
* <u>False Positive Rate</u>: 11.54%
* <u>False Negative Rate</u>: 3.57%
* <u>Recall</u>: 96.43%

### Inferences

1. <u>High Recall</u>: the ensemble architecture shows strong detection of synthetic audio
3. <u>False Positives</u>:
    * the system fails to identify 1 out of 8 tracks of Darth Vader's voice, probably owing to his deep and robotic speech
    * the system also fails to identify both tracks belonging to a human speaker, possibly because the speaker is not clearly audible in either of them
5. <u>False Negatives</u>:
    * the system fails to identify a WhatsApp voice note of an AI-generated speaker, possibly due to external interference or noise

### Limitations
1. Limited training dataset, consisting of mostly noise-less FLAC audio files
2. Limited testing dataset, mostly comprising noisy OPUS files that are heavily compressed
3. Rapid evolution of generative models may reduce detection reliability over time
4. False Positives could lead to denial of service, as real human speech would get identified as synthetic audio
5. False Negatives could lead to serious dangers such as identity theft, monetary scams, privacy invasion etc.
6. Model performance may vary across accents, speech styles and recording quality
7. Unethical usage could lead to large-scale monitoring and unauthorized profiling


## Conclusion

**This project demonstrates how deep learning can be applied to detect synthetic speech, while also highlighting the technical, ethical, and societal challenges involved in deploying such systems in real-world environments.**

**It positions AI systems not only as tools for detection, but as components within broader socio-technical systems requiring careful governance, evaluation, and responsible use. Future work in this field could include training over larger, more diverse datasets, improving model architecture and robustness, and integrating a real-time call monitoring pipeline.**
