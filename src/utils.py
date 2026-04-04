import torch
from datasets import Dataset

def build_scienceqa_prompt(question: str, choices: list) -> str:
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        return prompt

    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
        
    return prompt

def prepare_minicap_for_sft(raw_dataset, max_samples=None):
    """
    Cập nhật: Sử dụng .map() và .filter() để tránh bị timeout log.
    """
    SYSTEM_PROMPT = (
        "A conversation between The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
    )

    def format_sft_row(item):
        text_prompt = str(item.get("problem", ""))
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(item.get("solution", ""))}]
            }
        ]
        return {
            "messages": messages,
            "images": [item["image"]] 
        }

    dataset = raw_dataset.filter(lambda x: x.get("image") is not None)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    dataset = dataset.map(format_sft_row, num_proc=4)
    
    return dataset

def prepare_scienceqa_for_grpo(raw_dataset, processor=None, max_samples=None):
    labels = ["A", "B", "C", "D", "E"]

    SYSTEM_PROMPT = (
        "A conversation between The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    def format_row(item):
        text_prompt = build_scienceqa_prompt(item['question'], item['choices'])
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        correct_letter = labels[item['answer']]
        return {
            "prompt": messages,
            "images": [item['image']],  
            "ground_truth": correct_letter
        }

    dataset = raw_dataset.filter(lambda x: x['image'] is not None)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    dataset = dataset.map(format_row, num_proc=4) 
    return dataset