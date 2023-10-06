from pydantic import BaseModel


# Pydantic model for BertText, used for text classification
class BertText(BaseModel):
    # A dictionary containing story-related data
    story: dict


# Pydantic model for SummaryText, used for summarization
class SummaryText(BaseModel):
    # A dictionary containing data to be summarized
    data: dict


# Pydantic model for NerText, used for Named Entity Recognition (NER)
class NerText(BaseModel):
    # A list of text samples for NER processing
    text: list


class ArticleText(BaseModel):
    text: list