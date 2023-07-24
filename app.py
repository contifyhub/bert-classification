import logging
import secrets
import traceback
import os
from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from logging import config as logger_config

import config
from constants import (
    INDUSTRY_CLASSES, INDUSTRY_MAPPING,
    TOPIC_CLASSES, INDUSTRY_PREDICTION_THRESHOLD
)
from serializers import BertText, NerText
import numpy as np
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoConfig
import torch.neuron

from utils import clean_entity, extract_ner, predict_fn

num_cores = 0
os.environ['NEURON_RT_NUM_CORES'] = str(num_cores)

@lru_cache()
def get_settings():
    return config.BertClassifierSettings()


settings = get_settings()
logger_config.dictConfig(settings.LOGGING)


@lru_cache()
def get_bert_classifier():
    topic_neuron_model = torch.jit.load(f"{settings.INF_TOPIC_MODEL_FILE_NAME}")
    industry_neuron_model = torch.jit.load(f"{settings.INF_INDUSTRY_MODEL_FILE_NAME}")
    industry_tokenizer = AutoTokenizer.from_pretrained(
        f"{settings.INDUSTRY_MODEL_FILE_NAME}/")
    topic_tokenizer = AutoTokenizer.from_pretrained(
        f"{settings.TOPIC_MODEL_FILE_NAME}/")
    model_dir = f"{settings.NER_MODEL_DIR}"
    ner_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ner_neuron_model = torch.jit.load(
        os.path.join(model_dir, f"{settings.AWS_NEURON_TRACED_WEIGHTS_NAME}"))
    ner_model_config = AutoConfig.from_pretrained(model_dir)
    return industry_neuron_model, industry_tokenizer, topic_neuron_model, topic_tokenizer, ner_tokenizer, ner_neuron_model, ner_model_config




industry_neuron_model, industry_tokenizer, topic_neuron_model, topic_tokenizer, ner_tokenizer, ner_neuron_model, ner_model_config = get_bert_classifier()
app = FastAPI(docs_url=None, redoc_url=None)
security = HTTPBasic()

logger = logging.getLogger(__name__)


def is_authenticated_user(
        credentials: HTTPBasicCredentials = Depends(security),
        settings: config.BertClassifierSettings = Depends(get_settings)):
    correct_username = secrets.compare_digest(
        credentials.username, settings.BERT_CLASSIFICATION_USERNAME
    )
    correct_password = secrets.compare_digest(
        credentials.password, settings.BERT_CLASSIFICATION_PASSWORD
    )
    if not (correct_username and correct_password):
        logger.info(
            f"Authentication Failed: Incorrect: {credentials.username},"
            f" username or Password {credentials.password}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


@app.post('/predict/topic/')
async def predict_industry(story: BertText,
                        auth_status: int = Depends(is_authenticated_user)):
    """This api is used to tag Industry from Text.

    params: story: BertText
    Return: Tagged Entities
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_id = data['story_text'], data['story_id']
        max_length = 128
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
        logits = topic_neuron_model(*example_inputs_paraphrase)[0][0]
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        predicted_labels = [TOPIC_CLASSES[idx] for idx, label in
                            enumerate(predictions) if label == 1]
        output_labels = {'predicted_tags': predicted_labels, "story_id": story_id}
        logger.info(
            f"Topic Bert Classifier: completed prediction  for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} Seconds")
        return output_labels
    except Exception as err:
        logger.error(
            f"Topic Bert Classifier: Error occurred for story id :{story_id} "
            f" Error: {err} , Traceback: {traceback.format_exc()}")

@app.post('/predict/industry/')
async def predict_industry(story: BertText,
                        auth_status: int = Depends(is_authenticated_user)):
    """This api is used to tag Industry from Text.

    params: story: BertText
    Return: Tagged Entities
    """
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['story']
        input_text, story_id = data['story_text'], data['story_id']
        max_length = 128
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
        logits = industry_neuron_model(*example_inputs_paraphrase)[0][0]
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= INDUSTRY_PREDICTION_THRESHOLD)] = 1
        predicted_labels = [INDUSTRY_CLASSES[idx] for idx, label in
                            enumerate(predictions) if label == 1]
        industry_tags = [INDUSTRY_MAPPING[int(i)] for i in predicted_labels]
        output_labels = {'predicted_tags': industry_tags, "story_id": story_id}
        logger.info(
            f"Industry Bert Classifier: completed prediction  for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} Seconds")
        return output_labels
    except Exception as err:
        logger.error(
            f"Industry Bert Classifier: Error occurred for story id :{story_id} "
            f" Error: {err} , Traceback: {traceback.format_exc()}")


@app.post('/predict/ner/')
async def predict_ner(story: NerText,
                        auth_status: int = Depends(is_authenticated_user)):
    story_id = ""
    try:
        dt = datetime.now()
        data = story.dict()['text']
        input_text, story_id = data[0][0].lower(), data[0][1]['story_id']
        predictions = predict_fn(ner_tokenizer, ner_neuron_model, ner_model_config, input_text)
        # ner_results = extract_ner(predictions)
        logger.info(
            f"NER Bert Classifier: completed prediction  for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} Seconds")
        return predictions
    except Exception as err:
        logger.error(
            f"NER Bert : Error occurred for story id :{story_id} "
            f" Error: {err} , Traceback: {traceback.format_exc()}")




