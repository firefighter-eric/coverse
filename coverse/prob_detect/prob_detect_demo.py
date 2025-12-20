# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline
from pprint import pprint
from loguru import logger

text = "学习，就像嚼馒头，因为久了方觉甜"

model_path = "data/models/google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path, device_map="auto")

text_1 = "学习，就像[MASK]，因为久了方觉甜"

# %% count tokens

text_target = "嚼馒头"
input_ids = tokenizer(text_target, return_tensors="pt")["input_ids"][0][1:-1]
decoded_text = tokenizer.decode(input_ids)
logger.info(f"Input ids: {input_ids}, decoded text: {decoded_text}, len: {len(input_ids)}")

# %% pipeline

fill_mask_pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)
"""
https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/pipelines#transformers.FillMaskPipeline
"""

# %% fill mask
texts = [
    "学习，就像[MASK]，因为久了方觉甜",
    "学习，就像[MASK][MASK][MASK]，因为久了方觉甜",
]
for text in texts:
    results = fill_mask_pipeline(text, top_k=2)
    print("Input text:", text)
    pprint(results)
    print("-" * 100)

# target
targets = ["嚼", "馒", "头"]
texts = [
    "学习，就像[MASK][MASK][MASK]，因为久了方觉甜",
]
for text in texts:
    results = fill_mask_pipeline(text, targets=targets, top_k=2)
    print("Input text:", text)
    pprint(results)
    print("-" * 100)


# %% 

def detect_prob(text, target):
    target_token_ids = tokenizer(target)["input_ids"][1:-1]
    target_tokens = tokenizer.convert_ids_to_tokens(target_token_ids)
    logger.info(f"Target tokens: {target_tokens}")

    top_k = len(target_tokens)
    results = fill_mask_pipeline(text, targets=target_tokens, top_k=top_k)
    output = {'text': text, 'target': target, 'token_results': []}
    for i  in range(len(target_tokens)):
        target_token = target_tokens[i]
        result = results[i]

        pattern = ['[MASK]'] * len(target_tokens)
        pattern[i] = target_token
        pattern_str = " ".join(pattern)
        if pattern_str in result['sequence']:
            logger.info(f"Target token: {tokenizer.decode(target_token)}, "
                        f"Pattern: {pattern_str}, "
                        f"Score: {result['score']}")
            output['token_results'].append({
                    "token": target_token,
                    "pattern": pattern_str,
                    "score": result['score'],
            })
            break
    score = 1.0
    for token_res in output['token_results']:
        score *= token_res['score']
    output['final_score'] = score
            
                
    return output

text_1 = "学习，就像[MASK][MASK][MASK]，因为久了方觉甜"
text_2 = "学习，就像[MASK][MASK][MASK]，因为久了方觉甜"
target_1 = "嚼馒头"

print("Detecting probabilities:")
result_1 = detect_prob(text_1, target_1)
pprint(result_1)
result_2 = detect_prob(text_2, target_1)
pprint(result_2)
