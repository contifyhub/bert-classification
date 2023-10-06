# FastAPI NLP Classification API

A FastAPI-based API for performing various Natural Language Processing (NLP) tasks, including text classification, Named Entity Recognition (NER), and custom tagging. This API is designed to provide fast and efficient predictions using pre-trained BERT models.

## Features

- Text classification for industry, topic, business events, and custom tags.
- Named Entity Recognition (NER) for extracting entities from text.
- User authentication using HTTP Basic Authentication.
- Easy integration with pre-trained BERT models.
- Scalable and efficient with support for AWS Neuron acceleration.

## Prerequisites

Before getting started, make sure you have the following prerequisites installed:

- Python 3.9+
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/repo_name

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt

3. Configure your application settings in config.py.

4. Start the FastAPI application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

### Usage

**Authentication**

To access the API, you need to authenticate using HTTP Basic Authentication. Provide your username and password as specified in the configuration.

## Endpoints
- **/predict/topic/**: Predict topics from text.

- **/predict/industry/**: Tag industry entities from text.

- **/predict/ner/**: Perform Named Entity Recognition (NER) on text.

- **/predict/custom_tag/**: Tag custom tags from text for different clients.

- **/predict/business_event/**: Predict business events from text.

- **/predict/customtags/client_id**: Tag custom tags from text for different clients.

- **/predict/reject/**: Predict business and non-business from text.




