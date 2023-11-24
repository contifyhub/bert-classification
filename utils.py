import itertools
import torch
import numpy as np

from constants import MAX_LEN


def predict_fn(tokenizer, model, model_config, input_text):
    """
    Perform predictions using a tokenized input text and a model.

    Args:
        tokenizer: Tokenizer for tokenizing the input text.
        model: The neural model for making predictions.
        model_config: Configuration for the model.
        input_text (str): The input text to be tokenized and processed.

    Returns:
        list: A list of dictionaries containing prediction results, including tokens, labels, and offsets.
    """
    # Tokenize the input text and prepare input tensors
    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,  # Request offset mapping
    )
    inputs = (
        tokens["input_ids"],
        tokens["attention_mask"],
        tokens["token_type_ids"],
    )

    # Perform inference using the model
    with torch.inference_mode():
        logits = model(*inputs)["logits"]
        preds = np.argmax(logits, axis=2)

    outputs = []
    index = 0
    # Iterate through predictions and tokenize offsets
    # Ignore [CLS] (0th index) and [SEP] (last index)
    for label_id, token_id, offset in zip(
            preds.flatten()[1:-1], tokens.input_ids.flatten()[1:], tokens.offset_mapping.tolist()[0][1:]
    ):
        token = tokenizer.decode(token_id)
        label = model_config.id2label[label_id.item()]
        if "[PAD]" in token or label[2:] == "MISC":
            continue

        token_start, token_end = tuple(offset)  # Convert the 0-dimensional tensor to a tuple
        outputs.append(
            {
                "index": index,
                "token": token,
                "label": label,
                "start": token_start
            }
        )
        index += 1
    return outputs


def clean_entity(entity):
    """
        Clean and normalize extracted named entities.

        Args:
            entity (str): The extracted named entity.

        Returns:
            str: Cleaned and normalized entity.
        """
    return entity.replace('##', '')


def extract_ner(predictions):
    """
    Extract named entities from NER predictions.

    Args:
        predictions (list): List of dictionaries containing token information and labels.

    Returns:
        list: List of extracted named entities with labels and start indexes.
    """
    ner_results = []
    current_entity = None

    for token_data in predictions:
        token = token_data['token']
        label = token_data['label']

        # Check if a new entity starts (B- label)
        if label.startswith('B-') and "##" not in token:
            # If there's an existing entity, add it to the results
            if current_entity:
                current_entity['entity'] = clean_entity(current_entity['entity'])
                ner_results.append(current_entity)

            # Create a new entity
            current_entity = {
                'entity': token,
                'label': label[2:],
                'start_index': token_data['start']
            }
        elif label.startswith('B-') and "##" in token:
            # Continue appending tokens to the current entity
            if current_entity:
                current_entity['entity'] += token

        elif label.startswith('I-') and "##" not in token:
            # Append non-subword tokens to the current entity with a space
            if current_entity:
                current_entity['entity'] += ' ' + token

        elif label.startswith('I-') and "##" in token:
            # Continue appending tokens to the current entity
            if current_entity:
                current_entity['entity'] += token

        else:
            # Check if the current entity is complete and add it to the results
            if current_entity:
                current_entity['entity'] = clean_entity(current_entity['entity'])
                ner_results.append(current_entity)
                current_entity = None

    # Check if there's a pending entity after the loop and add it to the results
    if current_entity:
        current_entity['entity'] = clean_entity(current_entity['entity'])
        ner_results.append(current_entity)

    return ner_results


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



