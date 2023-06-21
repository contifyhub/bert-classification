import logging
import secrets
import traceback
from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from logging import config as logger_config
import config
from constants import INDUSTRY_MAPPING, INDUSTRY_PREDICTION_THRESHOLD
from serializers import BertText
import numpy as np
import torch
from functools import lru_cache
from transformers import (AutoModelForSequenceClassification,
                          Trainer,
                          AutoTokenizer)

@lru_cache()
def get_settings():
    return config.BertClassifierSettings()


settings = get_settings()
logger_config.dictConfig(settings.LOGGING)


@lru_cache()
def get_industry_bert_classifier():
    torch.cuda.empty_cache()
    industry_model = AutoModelForSequenceClassification.from_pretrained(
        f"{settings.INDUSTRY_MODEL_FILE_NAME}/")
    industry_trainer = Trainer(model=industry_model)
    industry_tokenizer = AutoTokenizer.from_pretrained(
        f"{settings.INDUSTRY_MODEL_FILE_NAME}/")
    return industry_model, industry_trainer, industry_tokenizer


model, trainer, tokenizer = get_industry_bert_classifier()
app = FastAPI(docs_url=None, redoc_url=None)
security = HTTPBasic()

logger = logging.getLogger(__name__)


def is_authenticated_user(
        credentials: HTTPBasicCredentials = Depends(security),
        settings: config.BertClassifierSettings = Depends(get_settings)):
    correct_username = secrets.compare_digest(
        credentials.username, settings.BERT_INDUSTRY_USERNAME
    )
    correct_password = secrets.compare_digest(
        credentials.password, settings.BERT_INDUSTRY_PASSWORD
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
        encoding = tokenizer(input_text, return_tensors="pt", truncation=True)
        encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}
        outputs = trainer.model(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= INDUSTRY_PREDICTION_THRESHOLD)] = 1
        industry_labels = [model.config.id2label[idx] for idx, label in
                       enumerate(predictions) if label == 1]
        industry_tags = [INDUSTRY_MAPPING[int(i)] for i in industry_labels]
        output_labels = {'predicted_tags': industry_tags, "story_id": story_id}
        logger.info(
            f"Industry Bert Classifier: completed prediction  for story_id: {story_id} "
            f"in {(datetime.now() - dt).total_seconds()} Seconds")
        return output_labels
    except Exception as err:
        logger.error(
            f"Industry Bert Classifier: Error occurred for story id :{story_id} "
            f" Error: {err} , Traceback: {traceback.format_exc()}")
