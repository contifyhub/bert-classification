from pydantic import BaseModel


class BertText(BaseModel):
    story: dict

