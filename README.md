# 🧠 EmotIQ – Emotion Intelligence

### [🚀 Try the Live Application](https://emotion-prediction-8ab6.onrender.com)

EmotIQ is a deep learning-based emotion classification system that analyzes text and identifies the emotion expressed by the writer. It uses NLP and Transformer-based models to classify text into six emotions: **Sadness, Joy, Love, Anger, Fear and Surprise.**

The project explores multiple deep learning architectures, compares their performance, and uses transfer learning with pretrained BERT to achieve accurate emotion classification.

## Key Features

- Six-class emotion classification.
- Implementation and comparison of RNN, LSTM, GRU and BiGRU.
- Fine-tuning pretrained BERT (`bert-base-uncased`).
- Text preprocessing and tokenization.
- Model evaluation using accuracy, precision, recall and F1-score.
- Real-time emotion prediction through a deployed web application.

## Dataset

**Dair-AI Emotion Dataset** – 20,000 labeled text samples.

| Split | Samples |
|---|---:|
| Training | 16,000 |
| Validation | 2,000 |
| Testing | 2,000 |

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Simple RNN | 26.45% |
| LSTM | 33.95% |
| GRU | 28.90% |
| BiGRU | 92.10% |
| **BERT (TensorFlow)** | **93.25%** |

## Key Findings

- BERT achieved the highest test accuracy of **93.25%**.
- BiGRU was the best-performing recurrent architecture with **92.10% accuracy**.
- Bidirectional learning significantly improved performance compared to the other recurrent models.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **NLP:** Hugging Face Transformers
- **Data Processing:** NumPy, Pandas
- **Evaluation:** Scikit-learn
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render

## Future Improvements

- Improve classification of ambiguous and context-dependent text.
- Experiment with additional Transformer architectures.
- Extend emotion detection to longer texts and conversations.

---

**EmotIQ – Turning Text into Emotion Intelligence.**
