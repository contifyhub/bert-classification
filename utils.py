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

def extract_ner(predictions):
    ner_results = []
    current_entity = None
    start_index = 0
    end_index = 0

    for token_data in predictions:
        token = token_data['token']
        label = token_data['label']

        if label.startswith('B-'):
            if current_entity:
                current_entity['end_index'] = end_index
                ner_results.append(current_entity)

            start_index = token_data['index']
            end_index = token_data['index'] + len(token)
            current_entity = {
                'entity': token,
                'label': label[2:],
                'start_index': start_index
            }
        elif label.startswith('I-'):
            if current_entity:
                current_entity['entity'] += ' ' + token
                end_index = token_data['index'] + len(token)
        else:
            if current_entity:
                current_entity['end_index'] = end_index
                ner_results.append(current_entity)
                current_entity = None

        end_index = token_data['index'] + len(token)

    if current_entity:
        current_entity['end_index'] = end_index
        ner_results.append(current_entity)

    return ner_results


def clean_entity(entity):
    return entity.replace(" ##", "")

