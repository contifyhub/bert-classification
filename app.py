import logging
import secrets
import json
import traceback
import os

from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from logging import config as logger_config

import config
from constants import (
    INDUSTRY_CLASSES, INDUSTRY_MAPPING,
    TOPIC_CLASSES, INDUSTRY_PREDICTION_THRESHOLD, SETTINGS, CUSTOM_TAG_BASE_PATH, CUSTOM_TAG_CLASSES,
    BUSINESS_EVENT_PREDICTION_THRESHOLD, CUSTOM_TAG_PREDICTION_THRESHOLD, TOPIC_PREDICTION_THRESHOLD,
    BUSINESS_EVENT_CLASSES
)
from serializers import BertText, NerText
import numpy as np
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoConfig
import torch.neuron

from utils import extract_ner, predict_fn

# Set the number of CPU cores to use for AWS Neuron
num_cores = 0  # 0 i.e. all cores,  max 4 core available for inf1
os.environ['NEURON_RT_NUM_CORES'] = str(num_cores)

# Configure logging using settings
logger_config.dictConfig(SETTINGS.LOGGING)

# Create a FastAPI application
app = FastAPI(docs_url=None, redoc_url=None)

# Create an instance of HTTPBasic for authentication
security = HTTPBasic()

# Initialize the logger
logger = logging.getLogger(__name__)


@lru_cache()
def get_settings():
    return config.BertClassifierSettings()

# Function to load custom tag models and binarizers
def load_custom_tag_models(client_id, model_dict):
    """
    Load custom tag models and binarizers for a specific client.

    Args:
        client_id (str): The ID of the client.
        model_dict (dict): A dictionary containing model and binarizer file paths.

    Returns:
        custom_tag_model (torch.jit.ScriptModule): The custom tag model.
        binarizer (AutoTokenizer): The custom tag binarizer (tokenizer).
    """
    model_file = model_dict.get('neuron_model')
    binarizer_file = model_dict.get('tokenizer')

    if not model_file:
        return None, None

    custom_tag_model = torch.jit.load(
        os.path.join(CUSTOM_TAG_BASE_PATH, os.path.join(str(client_id), model_file))
    )

    binarizer = None
    if binarizer_file:
        binarizer = AutoTokenizer.from_pretrained(
            os.path.join(CUSTOM_TAG_BASE_PATH, os.path.join(str(client_id), binarizer_file))
        )

    return custom_tag_model, binarizer


@lru_cache
def get_bert_classifier():
    """
    Load and configure various BERT-based classifiers and tokenizers.

    Returns:
        Tuple: A tuple containing the following elements in order:
        - industry_neuron_model (torch.jit.ScriptModule): The industry-specific BERT model.
        - industry_tokenizer (AutoTokenizer): The industry-specific BERT tokenizer.
        - topic_neuron_model (torch.jit.ScriptModule): The topic-specific BERT model.
        - topic_tokenizer (AutoTokenizer): The topic-specific BERT tokenizer.
        - ner_tokenizer (AutoTokenizer): The NER (Named Entity Recognition) tokenizer.
        - ner_neuron_model (torch.jit.ScriptModule): The NER model.
        - ner_model_config (AutoConfig): The configuration for the NER model.
        - business_event_neuron_model (torch.jit.ScriptModule): The business event-specific BERT model.
        - business_event_tokenizer (AutoTokenizer): The business event-specific BERT tokenizer.
        - custom_tag_model_map (dict): A dictionary mapping client IDs to custom tag models and binarizers.
    """
    # Load industry and topic models and tokenizers
    industry_neuron_model = torch.jit.load(f"{SETTINGS.INF_INDUSTRY_MODEL_FILE_NAME}")
    industry_tokenizer  = AutoTokenizer.from_pretrained(
        f"{SETTINGS.INDUSTRY_MODEL_FILE_NAME}/")

    # Load topic models and tokenizers
    topic_neuron_model = torch.jit.load(f"{SETTINGS.INF_TOPIC_MODEL_FILE_NAME}")
    topic_tokenizer =  AutoTokenizer.from_pretrained(
        f"{SETTINGS.TOPIC_MODEL_FILE_NAME}/")

    # Load NER models and tokenizers
    ner_model_dir = f"{SETTINGS.NER_MODEL_DIR}"
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_dir)
    ner_neuron_model = torch.jit.load(
        os.path.join(ner_model_dir, f"{SETTINGS.AWS_NEURON_TRACED_WEIGHTS_NAME}"))
    ner_model_config = AutoConfig.from_pretrained(ner_model_dir)

    # Load business event models and tokenizers
    business_event_neuron_model = torch.jit.load(f"{SETTINGS.INF_BUSINESS_EVENT_MODEL_FILE_NAME}")
    business_event_tokenizer = AutoTokenizer.from_pretrained(
        f"{SETTINGS.BUSINESS_EVENT_MODEL_FILE_NAME}/")

    # Create a defaultdict to store custom tag models
    custom_tag_model_map = {}
    # Loading all custom tag models and binarizers for different clients
    for client_id, model_dict in json.loads(SETTINGS.CUSTOM_TAG_CLIENT_MODEL_MAPPING).items():

        custom_tag_model, custom_tag_binarizer = load_custom_tag_models(client_id, model_dict)

        custom_tag_model_map[client_id] = {
            'model': custom_tag_model,
            'tokenizer': custom_tag_binarizer
        }

    return (industry_neuron_model, industry_tokenizer, topic_neuron_model, topic_tokenizer, ner_tokenizer,
            ner_neuron_model, ner_model_config, business_event_neuron_model, business_event_tokenizer,
            custom_tag_model_map)


(
    industry_neuron_model, industry_tokenizer, topic_neuron_model, topic_tokenizer, ner_tokenizer,
    ner_neuron_model, ner_model_config, business_event_neuron_model, business_event_tokenizer,
    custom_tag_model_map
) = get_bert_classifier()


def is_authenticated_user(
        credentials: HTTPBasicCredentials = Depends(security),
        settings: config.BertClassifierSettings = Depends(get_settings)
):
    """
    Function to authenticate a user based on HTTP Basic Authentication.

    Args:
        credentials (HTTPBasicCredentials): The user's credentials.
        settings (config.BertClassifierSettings): Application settings.

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """
    # Compare the provided username and password with the expected values
    correct_username = secrets.compare_digest(
        credentials.username, settings.BERT_CLASSIFICATION_USERNAME
    )
    correct_password = secrets.compare_digest(
        credentials.password, settings.BERT_CLASSIFICATION_PASSWORD
    )

    # If either the username or password is incorrect, raise an authentication error
    if not (correct_username and correct_password):
        logger.info(
            f"Authentication Failed: Incorrect: {credentials.username},"
            f" username or Password {credentials.password}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    # User is authenticated
    return True


@app.post('/predict/topic/')
async def predict_topic(story: BertText,
                        auth_status: int = Depends(is_authenticated_user)):
    """
    Endpoint for tagging topics from text.

    Args:
        story (BertText): The input text for prediction.
        auth_status (int, optional): Authentication status. Defaults to Depends(is_authenticated_user).

    Returns:
        dict: A dictionary containing predicted topic tags and story ID.
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_id = data['story_text'], data['story_id']
        max_length = 128

        # Tokenize the input text
        encoding = topic_tokenizer.encode_plus(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        example_inputs_paraphrase = (
            encoding["input_ids"],
            encoding["attention_mask"],
            encoding["token_type_ids"],
        )

        # Get logits from the topic_neuron_model
        logits = topic_neuron_model(*example_inputs_paraphrase)[0][0]
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= TOPIC_PREDICTION_THRESHOLD)] = 1

        # Map predicted labels to topic tags
        predicted_labels = [TOPIC_CLASSES[idx] for idx, label in
                            enumerate(predictions) if label == 1]
        output_labels = {'predicted_tags': predicted_labels, "story_id": story_id}

        # Log the prediction completion
        logger.info(
            f"Topic Bert Classifier: completed prediction for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return output_labels
    except Exception as err:
        # Log errors and exceptions
        logger.error(
            f"Topic Bert Classifier: Error occurred for story id: {story_id} "
            f"Error: {err}, Traceback: {traceback.format_exc()}")


@app.post('/predict/industry/')
async def predict_industry(story: BertText,
                           auth_status: int = Depends(is_authenticated_user)):
    """
    Endpoint for tagging Industry entities from text.

    Args:
        story (BertText): The input text for prediction.
        auth_status (int, optional): Authentication status. Defaults to Depends(is_authenticated_user).

    Returns:
        dict: A dictionary containing predicted Industry tags and story ID.
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_id = data['story_text'], data['story_id']
        max_length = 128

        # Tokenize the input text
        encoding = industry_tokenizer.encode_plus(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        example_inputs_paraphrase = (
            encoding["input_ids"],
            encoding["attention_mask"],
            encoding["token_type_ids"],
        )

        # Get logits from the industry_neuron_model
        logits = industry_neuron_model(*example_inputs_paraphrase)[0][0]
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= INDUSTRY_PREDICTION_THRESHOLD)] = 1

        # Map predicted labels to Industry tags
        predicted_labels = [INDUSTRY_CLASSES[idx] for idx, label in
                            enumerate(predictions) if label == 1]
        industry_tags = [INDUSTRY_MAPPING[int(i)] for i in predicted_labels]
        output_labels = {'predicted_tags': industry_tags, "story_id": story_id}

        # Log the prediction completion
        logger.info(
            f"Industry Bert Classifier: completed prediction for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return output_labels
    except Exception as err:
        # Log errors and exceptions
        logger.error(
            f"Industry Bert Classifier: Error occurred for story id: {story_id} "
            f"Error: {err}, Traceback: {traceback.format_exc()}")


@app.post('/predict/ner/')
async def predict_ner(story: NerText,
                      auth_status: int = Depends(is_authenticated_user)):
    """
    Endpoint for predicting Named Entities Recognition (NER) from text.

    Args:
        story (NerText): The input text for NER prediction.
        auth_status (int, optional): Authentication status. Defaults to Depends(is_authenticated_user).

    Returns:
        dict: A dictionary containing NER results and story ID.
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['text']
        input_text, story_id = data[0][0], data[0][1]['story_id']

        # Predict NER using the provided tokenizer, model, and configuration
        predictions = predict_fn(ner_tokenizer, ner_neuron_model, ner_model_config, input_text)
        ner_results = extract_ner(predictions)
        res = {
            'story_text': input_text,
            story_id: ner_results
        }

        # Log the NER prediction completion
        logger.info(
            f"NER Bert Classifier: completed prediction for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return res
    except Exception as err:
        # Log errors and exceptions
        logger.error(
            f"NER Bert Classifier: Error occurred for story id: {story_id} "
            f"Error: {err}, Traceback: {traceback.format_exc()}")


@app.post('/predict/custom_tag/')
async def predict_custom_tag(story: BertText,
                             auth_status: int = Depends(is_authenticated_user)):
    """
    Endpoint for tagging Custom Tags from text for different clients.

    Args:
        story (BertText): The input text for prediction.
        auth_status (int, optional): Authentication status. Defaults to Depends(is_authenticated_user).

    Returns:
        dict: A dictionary containing predicted custom tags and story ID.
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_id, client_id = data['story_text'], data['story_id'], data['client_id']
        max_length = 128

        # Tokenize the input text using the client-specific tokenizer
        encoding = custom_tag_model_map[str(client_id)]["tokenizer"].encode_plus(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        example_inputs_paraphrase = (
            encoding["input_ids"],
            encoding["attention_mask"],
            encoding["token_type_ids"],
        )

        # Get logits from the client-specific custom tag model
        logits = custom_tag_model_map[client_id]["model"](*example_inputs_paraphrase)[0][0]
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= CUSTOM_TAG_PREDICTION_THRESHOLD)] = 1

        # Map predicted labels to custom tags for the client
        predicted_labels = [CUSTOM_TAG_CLASSES[client_id][idx] for idx, label in
                            enumerate(predictions) if label == 1]
        output_labels = {'predicted_tags': predicted_labels, "story_id": story_id}

        # Log the prediction completion
        logger.info(
            f"Custom Tag Classifier: completed prediction for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return output_labels
    except Exception as err:
        # Log errors and exceptions
        logger.error(
            f"Custom Tag Classifier: Error occurred for story id: {story_id} "
            f"Error: {err}, Traceback: {traceback.format_exc()}")


@app.post('/predict/business_event/')
async def predict_business_event(story: BertText,
                                 auth_status: int = Depends(is_authenticated_user)):
    """
    Endpoint for predicting business events from text.

    Args:
        story (BertText): The input text for prediction.
        auth_status (int, optional): Authentication status. Defaults to Depends on(is_authenticated_user).

    Returns:
        dict: A dictionary containing predicted business event tags and story ID.
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_id = data['story_text'], data['story_id']
        max_length = 128

        # Tokenize the input text
        encoding = business_event_tokenizer.encode_plus(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        example_inputs_paraphrase = (
            encoding["input_ids"],
            encoding["attention_mask"],
            encoding["token_type_ids"],
        )

        # Get logits from the industry_neuron_model
        logits = business_event_neuron_model(*example_inputs_paraphrase)[0][0]
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= BUSINESS_EVENT_PREDICTION_THRESHOLD)] = 1

        # Map predicted labels to custom tags for the client
        predicted_labels = [BUSINESS_EVENT_CLASSES[idx] for idx, label in
                            enumerate(predictions) if label == 1]
        output_labels = {'predicted_tags': predicted_labels, "story_id": story_id}

        # Log the prediction completion
        logger.info(
            f"Business Events Classifier: completed prediction for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return output_labels
    except Exception as err:
        # Log errors and exceptions
        logger.error(
            f"Business Events Classifier: Error occurred for story id: {story_id} "
            f"Error: {err}, Traceback: {traceback.format_exc()}")



