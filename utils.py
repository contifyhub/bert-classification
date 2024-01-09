import itertools
import logging
import torch
import numpy as np
from enum import Enum
from constants import MAX_LEN, SETTINGS
from logging import config as logger_config
from typing import List, Optional, Tuple

# Configure logging using settings
logger_config.dictConfig(SETTINGS.LOGGING)

# Initialize the logger
logger = logging.getLogger(__name__)


class StrEnum(str, Enum):
    pass


class AggregationStrategy(StrEnum):
    """All the valid aggregation strategies for PostProcessing"""

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


class NER(object):
    def __init__(self, tokenizer, model, config, aggregation_strategy):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.aggregation_strategy = aggregation_strategy

    def __call__(self, input_text):
        self.input_text = input_text
        tokens = self.preprocess()
        model_outputs = self.forward(tokens)
        output = self.postprocess(model_outputs)
        result = self.clean_ner_result(output)
        return result

    def preprocess(self):
        tokens = self.tokenizer(
            self.input_text,
            return_tensors="pt",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,  # Request offset mapping
        )
        tokens["sentence"] = self.input_text
        tokens["is_last"] = True
        return tokens

    def forward(self, model_inputs):
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")
        inputs = (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["token_type_ids"],
        )
        with torch.inference_mode():
            logits = self.model(*inputs)

        return {
            "logits": logits["logits"] if isinstance(logits, dict)
            else logits[0],
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "is_last": is_last,
            **model_inputs
        }

    def postprocess(self, model_outputs, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        logits = model_outputs["logits"][0].numpy()
        sentence = model_outputs["sentence"]
        input_ids = model_outputs["input_ids"][0]
        offset_mapping = (
            model_outputs["offset_mapping"][0] if
            model_outputs["offset_mapping"] is not None else None
        )
        special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        input_ids = input_ids.numpy()
        offset_mapping = offset_mapping.numpy() if offset_mapping is not None else None

        pre_entities = self.gather_pre_entities(
            sentence, input_ids, scores, offset_mapping, special_tokens_mask
        )
        grouped_entities = self.aggregate(pre_entities)
        # Filter anything that is in ignore_labels
        entities = [
            entity
            for entity in grouped_entities
            if entity.get("entity", None) not in ignore_labels
            and entity.get("entity_group", None) not in ignore_labels
        ]
        return entities

    def gather_pre_entities(
            self,
            sentence: str,
            input_ids: np.ndarray,
            scores: np.ndarray,
            offset_mapping: Optional[List[Tuple[int, int]]],
            special_tokens_mask: np.ndarray,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    start_ind = start_ind.item()
                    end_ind = end_ind.item()
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer, "_tokenizer", None) and getattr(
                        self.tokenizer._tokenizer.model,
                        "continuing_subword_prefix",
                        None
                ):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    if self.aggregation_strategy in {
                        AggregationStrategy.FIRST,
                        AggregationStrategy.AVERAGE,
                        AggregationStrategy.MAX,
                    }:
                        logger.info(
                            "Tokenizer does not support real words, using fallback heuristic",
                            UserWarning,
                        )
                    is_subword = start_ind > 0 and " " not in sentence[
                                                              start_ind - 1: start_ind + 1]

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(self, pre_entities: List[dict]) -> List[dict]:
        if self.aggregation_strategy in {
            AggregationStrategy.NONE, AggregationStrategy.SIMPLE
        }:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities)

        if self.aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(
                self.group_sub_entities(entity_group_disagg))

        return entity_groups

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-", 1)[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def aggregate_words(self, entities: List[dict]) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        if self.aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError(
                "NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group))
                word_group = [entity]
        # Last item
        if word_group is not None:
            word_entities.append(
                self.aggregate_word(word_group))
        return word_entities

    def aggregate_word(self, entities: List[dict]) -> dict:
        word = self.tokenizer.convert_tokens_to_string(
            [entity["word"] for entity in entities])
        if self.aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif self.aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities,
                             key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif self.aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def clean_ner_result(self, ner_output):
        result_list = []
        for i in ner_output:
            if i['entity_group'] not in ['ORG', 'LOC', 'PER']:
                continue
            ner_dict = dict()
            ner_dict['entity'] = self.input_text[i['start']:i['end']]
            ner_dict['label'] = i['entity_group']
            ner_dict['start_index'] = i['start']
            result_list.append(ner_dict)
        return result_list


def predict_classes(model_map, text_list, multilabel=False):
    """This function is used to predict classes.

       params: model_map
               text_list
               multilabel:Default(False)
       Return: predictions
       """
    prediction = list()
    classification_models = model_map['models']
    binarizer_model = model_map.get('binarizer', [None])
    for classification_model, binarizer_model in zip(
            *[classification_models, binarizer_model]):
        pred = classification_model.predict(text_list)
        if multilabel:
            pred = binarizer_model.inverse_transform(pred)
        else:
            pred = pred.tolist()
        prediction.append(pred)
    if multilabel:
        final_predictions = [list(map(int, list(itertools.chain(*p))))
                             for p in zip(*prediction)]
    else:
        final_predictions = [list(pred_tup) for pred_tup in zip(*prediction)]
    return final_predictions
