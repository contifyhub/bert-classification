import torch
import numpy as np


def predict_fn(tokenizer, model, model_config, input_text):
    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    inputs = (
        tokens["input_ids"],
        tokens["attention_mask"],
        tokens["token_type_ids"],
    )
    with torch.inference_mode():
        logits = model(*inputs)["logits"]
        preds = np.argmax(logits, axis=2)

    entities = []
    current_entity = None
    current_start = None
    for i, label_id in enumerate(preds[0]):
        label = model_config.id2label[label_id.item()]
        token = tokenizer.decode(tokens.input_ids[0][i].item())
        if label.startswith('B-'):  # Start of an entity
            if current_entity:
                entities.append({
                    'entity': current_entity,
                    'score': 1.0,  # You may adjust the score accordingly based on your model
                    'index': i,
                    'word': tokenizer.decode(tokens.input_ids[0][current_start:i + 1].tolist()),
                    'start': tokens.word_to_tokens(current_start).start,
                    'end': tokens.word_to_tokens(i).end,
                })
            current_entity = label[2:]
            current_start = i
        elif label.startswith('I-'):  # Continuation of an entity
            if not current_entity:
                current_entity = label[2:]
                current_start = i

    # Append the last entity (if any)
    if current_entity:
        entities.append({
            'entity': current_entity,
            'score': 1.0,  # You may adjust the score accordingly based on your model
            'index': i,
            'word': tokenizer.decode(tokens.input_ids[0][current_start:i + 1].tolist()),
            'start': tokens.word_to_tokens(current_start).start,
            'end': tokens.word_to_tokens(i).end,
        })

    return entities

# def predict_fn(tokenizer,model,model_config, input_text):
#     tokens = tokenizer(
#         input_text,
#         return_tensors="pt",
#         max_length=512,
#         padding="max_length",
#         truncation=True,
#     )
#     inputs = (
#         tokens["input_ids"],
#         tokens["attention_mask"],
#         tokens["token_type_ids"],
#     )
#     with torch.inference_mode():
#         logits = model(*inputs)["logits"]
#         preds = np.argmax(logits, axis=2)
#
#     outputs = []
#     index = 0
#     # ignore [CLS] (0th index) and [SEP] (last index)
#     for label_id, token_id in zip(
#         preds.flatten()[1:-1], tokens.input_ids.flatten()[1:]
#     ):
#         if token_id == tokenizer.sep_token_id:
#             break
#         outputs.append(
#             {
#                 "index": index,
#                 "token": tokenizer.decode(token_id),
#                 "label": model_config.id2label[label_id.item()],
#             }
#         )
#         index += 1
#     return outputs

def clean_entity(entity):
    return entity.replace('##', '')

def extract_ner(predictions):
    ner_results = []
    current_entity = None

    for token_data in predictions:
        token = token_data['token']
        label = token_data['label']

        if label.startswith('B-') and "##" not in token:
            if current_entity:
                current_entity['entity'] = clean_entity(current_entity['entity'])
                ner_results.append(current_entity)

            current_entity = {
                'entity': token,
                'label': label[2:],
                'start_index': token_data['start']
            }
        elif label.startswith('B-') and "##" in token:
            if current_entity:
                current_entity['entity'] += token

        elif label.startswith('I-') and "##" not in token:
            if current_entity:
                current_entity['entity'] += ' ' + token

        elif label.startswith('I-') and "##" in token:
            if current_entity:
                current_entity['entity'] += token

        else:
            if current_entity:
                current_entity['entity'] = clean_entity(current_entity['entity'])
                ner_results.append(current_entity)
                current_entity = None

    if current_entity:
        current_entity['entity'] = clean_entity(current_entity['entity'])
        ner_results.append(current_entity)

    return ner_results


