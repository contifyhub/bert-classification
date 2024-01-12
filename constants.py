import config
import os
from functools import lru_cache
import json
from transformers import AutoConfig


ROOT_DIR = os.path.join(".")
BASE_PATH = "{}/ml_models/".format(ROOT_DIR)

# Import the settings from your configuration module
@lru_cache
def get_settings():
    # This function returns the settings defined in config.BertClassifierSettings
    return config.BertClassifierSettings()


# Call get_settings using lru_cache decorator to cache the result
SETTINGS = get_settings()

# Define prediction thresholds for different categories
# These thresholds are used for classification confidence level
# Adjust these thresholds based on your classification needs
INDUSTRY_PREDICTION_THRESHOLD = 0.7
TOPIC_PREDICTION_THRESHOLD = 0.7
CUSTOM_TAG_PREDICTION_THRESHOLD = 0.7
BUSINESS_EVENT_PREDICTION_THRESHOLD = 0.7
REJECT_PREDICTION_THRESHOLD = 0.85
GLOBAL_REJECT_PREDICTION_THRESHOLD = 0.85
# MAX_LEN is maximum length in number of tokens for the inputs to the
# transformer model. When the tokenizer is loaded with from_pretrained,
# this will be set to the value stored for the associated model
MAX_LEN = 512


# Define root and base paths
ROOT_DIR = os.path.join(".")
BERT_CUSTOM_TAG_BASE_PATH = "{}/ml_models/Custom_Tags/bert".format(ROOT_DIR)
BERT_REJECT_BASE_PATH = "{}/ml_models/Reject/bert".format(ROOT_DIR)
SK_LRN_CUSTOM_TAG_BASE_PATH = "{}/ml_models/Custom_Tags/sk-learn".format(ROOT_DIR)
REJECT_TAG_BASE_PATH = "{}/ml_models/".format(ROOT_DIR)

BUSINESS_EVENT_MAPPING = {
 98485: 2,
 98495: 3,
 98496: 4,
 98498: 5,
 98499: 6,
 98500: 7,
 98501: 8,
 98502: 9,
 98503: 10,
 98504: 11,
 98505: 12,
 98506: 14,
 98559: 13,
 98486: 40,
 98540: 41,
 98541: 42,
 98542: 43,
 98487: 44,
 98543: 45,
 98544: 46,
 98545: 47,
 98488: 15,
 98509: 16,
 98510: 17,
 98511: 18,
 98512: 19,
 98513: 20,
 98515: 21,
 98489: 37,
 98533: 38,
 98536: 39,
 98491: 36,
 98490: 31,
 98529: 32,
 98530: 33,
 98531: 34,
 98532: 35,
 98518: 22,
 98519: 23,
 98520: 24,
 98521: 25,
 98523: 26,
 98524: 27,
 98525: 28,
 98526: 29,
 98527: 30,
 98546: 48,
 98547: 49,
 98548: 50,
 98549: 51,
 98550: 52,
 98551: 53,
 98552: 54,
 98553: 55,
 98555: 56
}

# contify's custom industry to standard industry mapping
INDUSTRY_MAPPING = {
 98636: 622,
 98637: 545,
 98638: 128,
 98639: 548,
 98640: 480,
 98645: 483,
 98646: 145,
 98647: 254,
 98648: 484,
 98649: 626,
 98650: 572,
 98651: 279,
 98653: 627,
 98654: 350,
 98655: 632,
 98656: 633,
 98657: 628,
 98658: 15,
 98659: 634,
 98660: 247,
 98661: 635,
 98662: 249,
 98663: 50,
 98664: 617,
 98665: 164,
 98666: 284,
 98667: 605,
 98668: 383,
 98669: 127,
 98670: 106,
 98671: 618,
 98673: 329,
 98674: 619,
 98675: 472,
 98676: 523,
 98678: 620,
 98680: 636,
 98681: 538,
 98682: 476,
 98683: 474,
 98684: 539,
 98685: 475,
 98686: 229,
 98687: 621,
 98688: 630,
 98689: 281,
 98690: 97,
 98691: 637,
 98692: 69,
 98693: 222,
 98695: 542,
 98696: 511,
 98697: 258,
 98698: 510,
 98699: 465,
 98700: 615,
 98706: 185,
 98707: 324,
 98708: 322,
 98709: 360,
 98710: 481,
 98711: 643,
 98713: 639,
 98716: 640,
 98718: 482,
 98719: 12,
 98720: 121,
 98722: 641,
 98723: 642,
 98724: 644,
 98725: 368,
 98726: 566,
 98728: 569,
 98729: 374,
 98732: 466,
 98733: 645,
 98734: 646,
 98735: 66,
 98738: 648,
 98739: 609,
 98740: 4,
 98741: 295,
 98742: 495,
 98743: 496,
 98744: 305,
 98745: 638,
 98746: 497,
 98747: 271,
 98748: 500,
 98749: 501,
 98750: 502,
 98751: 503,
 98752: 504,
 98753: 147,
 98754: 340,
 98755: 506,
 98756: 507,
 98757: 508,
 98758: 144,
 98759: 126,
 98760: 629,
 98761: 584,
 98762: 491,
 98763: 586,
 98641: 551,
 98672: 470,
 98677: 394,
 98714: 375,
 98727: 369}


INDUSTRY_CLASSES = list(AutoConfig.from_pretrained(os.path.join(BASE_PATH, SETTINGS.INDUSTRY_MODEL_FILE_NAME)).id2label.values())
TOPIC_CLASSES = list(AutoConfig.from_pretrained(os.path.join(BASE_PATH, SETTINGS.TOPIC_MODEL_FILE_NAME)).id2label.values())
BUSINESS_EVENT_CLASSES = list(AutoConfig.from_pretrained(os.path.join(BASE_PATH, SETTINGS.BUSINESS_EVENT_MODEL_FILE_NAME)).id2label.values())


CUSTOM_TAG_CLASSES = {
    214: list(AutoConfig.from_pretrained(os.path.join(
     BERT_CUSTOM_TAG_BASE_PATH,
     os.path.join(str(214), json.loads(SETTINGS.CUSTOM_TAG_CLIENT_MODEL_MAPPING)[str(214)]['tokenizer'])
    )
    ).id2label.values())
}

PREDICTION_TO_STORY_STATUS_MAPPING = {
 0: 2,
 1: -1
}
PREDICTION_TO_STORY_GLOBAL_TAG_MAPPING = {0: 17012, 1: 16727, 2: 16718}

CLASSIFIED_MODELS = {
    'Reject': [
        {'model_file': 'SI_Non_Business_Reject.pkl'}
    ],
    '135': [
        {
            'model_file': 'custom_tags_model_135_en_1.pkl',
            'binarizer_file': 'custom_tags_binarizer_135_en_1.pkl'
        }
    ]
}

CONTIFY_FOR_SALES_COMPANY_PREFERENCE_ID = 82

