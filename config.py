from pydantic import BaseSettings
import os

ROOT_DIR = os.path.join("..")


class BertClassifierSettings(BaseSettings):
    """
    Settings class for BertClassifier configuration.
    NOTE:  File name starts with INF is only compatible with AWS INF instances.
    """
    BERT_CLASSIFICATION_USERNAME: str = ''
    BERT_CLASSIFICATION_PASSWORD: str = ''

    BERT_INDUSTRY_USERNAME: str = ''
    BERT_INDUSTRY_PASSWORD: str = ''

    TOPIC_MODEL_FILE_NAME = ''  # Path to the topic model file
    INF_TOPIC_MODEL_FILE_NAME = ''  # Path to the inferred topic model file

    INDUSTRY_MODEL_FILE_NAME = ''  # Path to the industry model file
    INF_INDUSTRY_MODEL_FILE_NAME = ''  # Path to the inferred industry model file

    BUSINESS_EVENT_MODEL_FILE_NAME = ''  # Path to the business event model file
    INF_BUSINESS_EVENT_MODEL_FILE_NAME = ''  # Path to the inferred business event model file

    SUMMARY_MODEL_FILE_NAME = ''  # Path to the summary model file

    AWS_NEURON_TRACED_WEIGHTS_NAME = ''  # Name of AWS Neuron traced weights
    NER_MODEL_DIR = ''  # Directory containing NER models

    CUSTOM_TAG_CLIENT_MODEL_MAPPING = ''  # Mapping for custom tag client models

    CLIENT_STORY_REJECT_MODEL_MAPPING = ''  # Mapping for custom tag client models

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
        env_file = ".env"  # Load environment variables from .env file


