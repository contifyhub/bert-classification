from pydantic import BaseModel


class BertText(BaseModel):
    story: dict


class SummaryText(BaseModel):
    data: dict

class NerText(BaseModel):
    text: list
