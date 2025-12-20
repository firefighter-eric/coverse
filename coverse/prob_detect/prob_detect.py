
from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline
from pprint import pprint
from loguru import logger
import json
import numpy as np


model_path = "data/models/google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path, device_map="auto")
fill_mask_pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

# %% 

def detect_one_token_prob(text: str, target: str, index: int = 0):
    results = fill_mask_pipeline(text, targets=[target], top_k=1)
    result = results[index][0]
    return result


def detect_prob(text, target):
    target_token_ids = tokenizer(target)["input_ids"][1:-1]
    target_tokens = [tokenizer.decode([token_id]) for token_id in target_token_ids]
    logger.info(f"Target tokens: {target_tokens}")

    output = {'text': text, 'target': target, 'token_results': []}
    for i in range(len(target_tokens)):
        target_token = target_tokens[i]
        result = detect_one_token_prob(text, target=target_token, index=i)
        output['token_results'].append(result)

    prob = 1.0
    log_prob = 0.0
    for token_res in output['token_results']:
        prob *= token_res['score']
        log_prob += -1.0 * np.log(token_res['score'] + 1e-12)
    output['prob'] = prob
    output['log_prob'] = log_prob
                
    return output

texts = ["学习，就像[MASK][MASK][MASK]，因为久了方觉甜",
          "学习，就像[MASK][MASK][MASK]",
         "学习，就像[MASK][MASK][MASK]，因为都很困难",]

target = "嚼馒头"

outputs = []
print("Detecting probabilities:")
for text in texts:
    result = detect_prob(text, target)
    pprint(result)
    print("-" * 100)
    outputs.append(result)

# summary
for output in outputs:
    print(f"Prob: {output['prob']:.2e} | Log Prob: {output['log_prob']:.2f} | Text: {output['text']} | Target: {output['target']}")


with open("data/prob_detect/test_1/2.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)
