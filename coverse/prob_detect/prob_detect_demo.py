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
input_ids = tokenizer(text_target)["input_ids"][1:-1]
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
