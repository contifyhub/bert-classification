import torch
import numpy as np


def predict_fn(tokenizer,model,model_config, input_text):
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

    outputs = []
    index = 0
    # ignore [CLS] (0th index) and [SEP] (last index)
    for label_id, token_id in zip(
        preds.flatten()[1:-1], tokens.input_ids.flatten()[1:]
    ):
        if token_id == tokenizer.sep_token_id:
            break
        outputs.append(
            {
                "index": index,
                "token": tokenizer.decode(token_id),
                "label": model_config.id2label[label_id.item()],
            }
        )
        index += 1
    return outputs

def clean_entity(entity):
    return entity.replace('##', '')

def extract_ner(predictions):
    ner_results = []
    current_entity = None

    for token_data in predictions:
        token = token_data['word']
        label = token_data['entity']

        if label.startswith('B-') and "##" not in token:
            if current_entity:
                current_entity['entity'] = clean_entity(current_entity['entity'])
                ner_results.append(current_entity)

            current_entity = {
                'entity': token,
                'label': label[2:],
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


