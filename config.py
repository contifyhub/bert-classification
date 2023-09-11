from pydantic import BaseSettings
import os

ROOT_DIR = os.path.join("..")


class BertClassifierSettings(BaseSettings):
    BERT_CLASSIFICATION_USERNAME: str = ''
    BERT_CLASSIFICATION_PASSWORD: str = ''
    BERT_INDUSTRY_USERNAME: str = ''
    BERT_INDUSTRY_PASSWORD: str = ''
    TOPIC_MODEL_FILE_NAME = ''
    SUMMARY_MODEL_FILE_NAME = ''
    INDUSTRY_MODEL_FILE_NAME = ''
    INF_TOPIC_MODEL_FILE_NAME = ''
    INF_INDUSTRY_MODEL_FILE_NAME = ''
    AWS_NEURON_TRACED_WEIGHTS_NAME = ''
    NER_MODEL_DIR = ''
    AWS_NEURON_C_TRACED_WEIGHTS_NAME_13 = ''
    CUSTOM_TAG_MODEL_DIR = ''
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'file_handler': {
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': 'logs/cfy_bert_classification.log',
                'level': 'DEBUG',
                'formatter': 'simple',
                'when': 'midnight',
                'interval': 1,
                'backupCount': 7
            },
        },
        'loggers': {
            '1': {
                'handlers': ['file_handler'],
                'level': 'DEBUG',
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['file_handler'],
        },
    }

    class Config:
        env_file = ".env"