import logging
import secrets
import json
import traceback
import os

from datetime import datetime
from collections import defaultdict

import joblib
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from logging import config as logger_config

import config
from constants import (
    INDUSTRY_CLASSES, INDUSTRY_MAPPING,
    TOPIC_CLASSES, INDUSTRY_PREDICTION_THRESHOLD, SETTINGS,
    BERT_CUSTOM_TAG_BASE_PATH, CUSTOM_TAG_CLASSES,
    BUSINESS_EVENT_PREDICTION_THRESHOLD, CUSTOM_TAG_PREDICTION_THRESHOLD,
    TOPIC_PREDICTION_THRESHOLD,
    BUSINESS_EVENT_CLASSES, BUSINESS_EVENT_MAPPING, CLASSIFIED_MODELS,
    BASE_PATH, SK_LRN_CUSTOM_TAG_BASE_PATH,
    REJECT_TAG_BASE_PATH, BERT_REJECT_BASE_PATH, REJECT_PREDICTION_THRESHOLD,
    PREDICTION_TO_STORY_STATUS_MAPPING,
    CONTIFY_FOR_SALES_COMPANY_PREFERENCE_ID,
    GLOBAL_REJECT_PREDICTION_THRESHOLD, PREDICTION_TO_STORY_GLOBAL_TAG_MAPPING
)
from serializers import BertText, NerText, ArticleText
import numpy as np
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoConfig
import torch.neuron

from utils import predict_classes, NER

from typing import List, Optional, Tuple

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
        os.path.join(BERT_CUSTOM_TAG_BASE_PATH, os.path.join(str(client_id), model_file))
    )

    binarizer = None
    if binarizer_file:
        binarizer = AutoTokenizer.from_pretrained(
            os.path.join(BERT_CUSTOM_TAG_BASE_PATH, os.path.join(str(client_id), binarizer_file))
        )

    return custom_tag_model, binarizer


# Function to load reject models and binarizers
def load_reject_models(client_id, model_dict):
    """
    Load Reject models and binarizers.

    Args:
        client_id (str): The ID of the client.
        model_dict (dict): A dictionary containing model and binarizer file paths.

    Returns:
        reject_model (torch.jit.ScriptModule): The Reject model.
        binarizer (AutoTokenizer): The Reject binarizer (tokenizer).
    """
    model_file = model_dict.get('neuron_model')
    binarizer_file = model_dict.get('tokenizer')

    if not model_file:
        return None, None

    reject_model = torch.jit.load(
        os.path.join(BERT_REJECT_BASE_PATH, os.path.join(str(client_id), model_file))
    )

    binarizer = None
    if binarizer_file:
        binarizer = AutoTokenizer.from_pretrained(
            os.path.join(BERT_REJECT_BASE_PATH, os.path.join(str(client_id), binarizer_file))
        )

    return reject_model, binarizer


def get_ml_classifier():
    classified_model_map = defaultdict(lambda: defaultdict(list))
    for model_type in CLASSIFIED_MODELS:
        base_path = SK_LRN_CUSTOM_TAG_BASE_PATH
        if model_type == "Reject":
            base_path = REJECT_TAG_BASE_PATH
        for model_dict in CLASSIFIED_MODELS[model_type]:
            model_file = model_dict.get('model_file')
            if not model_file:
                continue
            classified_model_map[model_type]['models'].append(joblib.load(
                os.path.join(base_path, os.path.join(model_type, model_file))
            ))

            binarizer_file = model_dict.get('binarizer_file')
            if binarizer_file:
                classified_model_map[model_type]['binarizer'].append(joblib.load(
                    os.path.join(base_path,
                                 os.path.join(model_type, binarizer_file))
                ))


    return classified_model_map


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
    industry_neuron_model = torch.jit.load(f"{os.path.join(BASE_PATH, SETTINGS.INF_INDUSTRY_MODEL_FILE_NAME)}")
    industry_tokenizer = AutoTokenizer.from_pretrained(f"{os.path.join(BASE_PATH, SETTINGS.INDUSTRY_MODEL_FILE_NAME)}/")

    # Load topic models and tokenizers
    topic_neuron_model = torch.jit.load(f"{os.path.join(BASE_PATH, SETTINGS.INF_TOPIC_MODEL_FILE_NAME)}")
    topic_tokenizer = AutoTokenizer.from_pretrained(f"{os.path.join(BASE_PATH, SETTINGS.TOPIC_MODEL_FILE_NAME)}/")

    # Load NER models and tokenizers
    ner_model_dir = f"{os.path.join(BASE_PATH, SETTINGS.NER_MODEL_DIR)}"
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_dir)
    ner_neuron_model = torch.jit.load(os.path.join(ner_model_dir, f"{SETTINGS.AWS_NEURON_TRACED_WEIGHTS_NAME}"))
    ner_model_config = AutoConfig.from_pretrained(ner_model_dir)

    # Load business event models and tokenizers
    business_event_neuron_model = torch.jit.load(f"{os.path.join(BASE_PATH, SETTINGS.INF_BUSINESS_EVENT_MODEL_FILE_NAME)}")
    business_event_tokenizer = AutoTokenizer.from_pretrained(
        f"{os.path.join(BASE_PATH, SETTINGS.BUSINESS_EVENT_MODEL_FILE_NAME)}/")

    # Create a defaultdict to store custom tag models
    custom_tag_model_map = {}
    # Loading all custom tag models and binarizers for different clients
    for client_id, model_dict in json.loads(SETTINGS.CUSTOM_TAG_CLIENT_MODEL_MAPPING).items():
        custom_tag_model, custom_tag_binarizer = load_custom_tag_models(client_id, model_dict)

        custom_tag_model_map[client_id] = {
            'model': custom_tag_model,
            'tokenizer': custom_tag_binarizer
        }

    # Create a defaultdict to store reject models
    reject_model_map = {}
    # Loading reject models and tokenizer for different clients
    for client_id, model_dict in json.loads(
            SETTINGS.CLIENT_STORY_REJECT_MODEL_MAPPING).items():
        reject_model, reject_model_tokenizer = load_reject_models(
            client_id, model_dict)

        reject_model_map[client_id] = {
            'model': reject_model,
            'tokenizer': reject_model_tokenizer
        }

    return (industry_neuron_model, industry_tokenizer, topic_neuron_model, topic_tokenizer, ner_tokenizer,
            ner_neuron_model, ner_model_config, business_event_neuron_model, business_event_tokenizer,
            custom_tag_model_map, reject_model_map)


(
    industry_neuron_model, industry_tokenizer, topic_neuron_model, topic_tokenizer, ner_tokenizer,
    ner_neuron_model, ner_model_config, business_event_neuron_model, business_event_tokenizer,
    custom_tag_model_map, reject_model_map
) = get_bert_classifier()

classified_model_map = get_ml_classifier()


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
        ner = NER(ner_tokenizer, ner_neuron_model, ner_model_config, aggregation_strategy='max')
        ner_output = ner(input_text)
        res = {
            'story_text': input_text,
            story_id: ner_output
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
        logits = custom_tag_model_map[str(client_id)]["model"](*example_inputs_paraphrase)[0][0]
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


@app.post('/predict/reject_by_client_id/')
async def predict_reject_by_client_id(story: BertText,
                             auth_status: int = Depends(is_authenticated_user)):
    """
    Endpoint for Rejecting stories for different clients.

    Args:
        story (BertText): The input text for prediction.
        auth_status (int, optional): Authentication status. Defaults to Depends(is_authenticated_user).

    Returns:
        dict: A dictionary containing predicted custom tags and story ID.
    """
    story_uuid = ""
    client_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_uuid, client_id = data['story_text'], \
            data['story_uuid'], data['client_id']
        max_length = 512

        # Tokenize the input text using the client-specific tokenizer
        encoding = reject_model_map[str(client_id)]["tokenizer"].encode_plus(
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

        # Get logits from the client-specific reject model
        logits = reject_model_map[str(client_id)]["model"](*example_inputs_paraphrase)[0]
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        if client_id == CONTIFY_FOR_SALES_COMPANY_PREFERENCE_ID:
            predicted_tag = torch.argmax(probabilities).item()
            probability, _ = torch.max(probabilities, dim=1)
            max_probabilities = probability.item()
            if max_probabilities > GLOBAL_REJECT_PREDICTION_THRESHOLD:
                predicted_class = 1
            else:
                predicted_class = 0
        else:
            max_probabilities = probabilities[0][1].item()
            if max_probabilities > REJECT_PREDICTION_THRESHOLD:
                predicted_class = 1
            else:
                predicted_class = 0

        # Map predicted status to story status
        predicted_status = PREDICTION_TO_STORY_STATUS_MAPPING[predicted_class]
        if client_id == CONTIFY_FOR_SALES_COMPANY_PREFERENCE_ID:
            output_labels = {'predicted_status': predicted_status,
                             'predicted_tag': PREDICTION_TO_STORY_GLOBAL_TAG_MAPPING[predicted_tag],
                             "story_uuid": story_uuid}
        else:
            output_labels = {'predicted_status': predicted_status, "story_uuid": story_uuid}

        # Log the prediction completion
        logger.info(
            f"Rejection Model: completed prediction for story_id: {story_uuid} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return output_labels
    except Exception as err:
        # Log errors and exceptions
        logger.error(
            f"Rejection Model: Error occurred for client_id: {client_id} "
            f"story id: {story_uuid} Error: {err}, Traceback: {traceback.format_exc()}")


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
        business_event_tags = [int(i) for i in predicted_labels]
        output_labels = {'predicted_tags': business_event_tags, "story_id": story_id}

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


@app.post('/predict/customtags/{client_id}/')
async def predict_custom_tags(client_id: str, story: ArticleText,
                              auth_status: int = Depends(is_authenticated_user)):
    """This api is used to attach Customtags to ArticleText.

    params: story: ArticleText
            client_id: id of the client
    Return: predictions
    """
    try:
        dt = datetime.now()
        data = story.dict()
        text_list = data['text']
        ct_models_list = classified_model_map[client_id]
        predictions = predict_classes(ct_models_list, text_list, multilabel=True)
        logger.info(
            f"Custom Tag  Classifier: completed prediction for client_id: {client_id} "
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return {
            'result': predictions
        }
    except Exception as err:
        logger.error(
            f"Custom Tag Classifier: Error occurred for client_id: {client_id} "
            f"Error: {err}, Traceback: {traceback.format_exc()}")


@app.post('/predict/reject/')
async def predict_reject(story: ArticleText,
                         auth_status: int = Depends(is_authenticated_user)):
    """This api is used to  reject ArticleText.

    params: story: ArticleText
    Return: predictions
    """
    try:
        dt = datetime.now()
        data = story.dict()
        text_list = data['text']
        ct_models_list = classified_model_map['Reject']
        predictions = predict_classes(ct_models_list, text_list)
        logger.info(
            f"Reject: completed prediction for rejection"
            f"in {(datetime.now() - dt).total_seconds()} seconds")
        return {
            'result': predictions
        }
    except Exception as err:
        logger.error(
            f"Reject: Error occurred for Rejection "
            f"Error: {err}, Traceback: {traceback.format_exc()}")
