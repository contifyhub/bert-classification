import logging
import secrets
import traceback
from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from logging import config as logger_config
import config
from constants import INDUSTRY_MAPPING, INDUSTRY_PREDICTION_THRESHOLD
from serializers import BertText, SummaryText
import numpy as np
import torch
from functools import lru_cache
from transformers import (AutoModelForSequenceClassification,
                          Trainer,
                          AutoTokenizer, AutoModelForSeq2SeqLM)

@lru_cache()
def get_settings():
    return config.BertClassifierSettings()


settings = get_settings()
logger_config.dictConfig(settings.LOGGING)





app = FastAPI(docs_url=None, redoc_url=None)
security = HTTPBasic()

logger = logging.getLogger(__name__)

@lru_cache()
def get_bert_summarizer():
    summary_tokenizer = AutoTokenizer.from_pretrained(
        f"./{settings.SUMMARY_MODEL_FILE_NAME}"
    )
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(
        f"./{settings.SUMMARY_MODEL_FILE_NAME}"
        f"")
    return summary_tokenizer, summary_model


summarization_tokenizer, summarization_model = get_bert_summarizer()

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

@app.post('/predict/summarize_text/')
async def summarize_text(story: SummaryText,
                         auth_status: int = Depends(is_authenticated_user)):
    """This tes-api is used to tag ner from Text.
    params: story: SummaryText
    Return: Summarized text
    """
    summary_text = ""
    try:
        data = story.dict()
        data_dict = data['data']
        summary_text = data_dict["text"]
        max_len = data_dict["max_len"]
        no_of_beams = data_dict["no_of_beams"]
        inputs = summarization_tokenizer(
            [summary_text],
            max_length=1024,
            return_tensors="pt",
            truncation=True
        )
        # Generate Summary
        summary_ids = summarization_model.generate(inputs["input_ids"],
                                                   num_beams=no_of_beams,
                                                   min_length=0,
                                                   max_length=max_len)
        bert_output = summarization_tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
            )
        return bert_output[0]

    except Exception as err:
        logger.info(f"Ner Bert: Error occurred for story {summary_text} "
                    f" Error: {err} , Traceback: {traceback.format_exc()}")

