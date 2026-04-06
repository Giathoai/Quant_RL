import sys
import os
import torch
import io
import gc
import re
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
import numpy as np
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

def get_sqa_category(row):
    subject = str(row.get('subject', '')).lower()
    if 'natural' in subject: subj = 'NAT'
    elif 'social' in subject: subj = 'SOC'
    elif 'language' in subject: subj = 'LAN'
    else: subj = 'OTH'

    has_image = 'image' in row and pd.notna(row['image'])
    has_text = 'hint' in row and pd.notna(row['hint']) and str(row['hint']).strip() != ""
    if has_image: ctx = 'IMG'
    elif has_text: ctx = 'TXT'
    else: ctx = 'NO'

    grade_str = str(row.get('grade', ''))
    match = re.search(r'\d+', grade_str)
    if match:
        grade_num = int(match.group(0))
        grd = 'G1-6' if grade_num <= 6 else 'G7-12'
    else:
        grd = 'UNK'

    return subj, ctx, grd

def evaluate_model(model_path, df, lora_path=None):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        
    processor_path = lora_path if lora_path else model_path
    processor = AutoProcessor.from_pretrained(processor_path)
    
    model.eval()

    correct = 0
    img_correct = 0
    img_total = 0
    predictions = []
    
    metrics = {
        'subject': {'NAT': [0, 0], 'SOC': [0, 0], 'LAN': [0, 0], 'OTH': [0, 0]},
        'context': {'IMG': [0, 0], 'TXT': [0, 0], 'NO': [0, 0]},
        'grade': {'G1-6': [0, 0], 'G7-12': [0, 0], 'UNK': [0, 0]}
    }

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {os.path.basename(model_path)}"):
            choices_str = ""
            labels = ["A", "B", "C", "D", "E"]
            if isinstance(row['choices'], list) or isinstance(row['choices'], np.ndarray):
                for i, c in enumerate(row['choices']):
                    choices_str += f"{labels[i]}. {c}\n"
            else:
                choices_str = str(row['choices'])
                
            text_content = (
                f"{row['question']}\n\nChoices:\n{choices_str}\n"
                "You are a helpful assistant. Answer the user's question based on the image provided. "
                "Output your thinking process within the <think> and </think> tags. "
                "Once the final answer is confirmed, put it within <answer> and </answer>."
            )
            
            content = [{"type": "text", "text": text_content}]
            
            if 'image' in row and pd.notna(row['image']):
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img_data = Image.open(io.BytesIO(img_data['bytes']))
                content.insert(0, {"type": "image", "image": img_data})

            messages = [{"role": "user", "content": content}]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=768)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            prediction = output_text.strip().upper()
            target_idx = int(row['answer'])
            target = chr(ord('A') + target_idx)
            
            match = re.search(r'<answer>\s*([A-E])\s*</answer>', prediction, re.IGNORECASE)
            if match:
                extracted_pred = match.group(1).upper()
            else:
                extracted_pred = prediction 
                
            is_correct = 1 if target in extracted_pred else 0
            correct += is_correct
            
            has_image = 'image' in row and pd.notna(row['image'])
            if has_image:
                img_total += 1
                img_correct += is_correct
            
            subj, ctx, grd = get_sqa_category(row)
            metrics['subject'][subj][0] += is_correct
            metrics['subject'][subj][1] += 1
            metrics['context'][ctx][0] += is_correct
            metrics['context'][ctx][1] += 1
            metrics['grade'][grd][0] += is_correct
            metrics['grade'][grd][1] += 1
            
            predictions.append(prediction)

    accuracy = (correct / len(df)) * 100
    img_accuracy = (img_correct / img_total * 100) if img_total > 0 else 0.0
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy, img_accuracy, metrics, predictions

def print_detailed_metrics(name, acc, img_acc, metrics):
    print(f"\n" + "="*50)
    print(f" THỐNG KÊ CHI TIẾT: {name.upper()}")
    print("="*50)
    print(f"  Accuracy (Overall) : {acc:.2f}%")
    print(f"  IMG-Accuracy       : {img_acc:.2f}%")
    
    print("\n Phân tích theo Môn học (Subject):")
    for k, v in metrics['subject'].items():
        if v[1] > 0: print(f"  - {k:<5} : {v[0]/v[1]*100:>5.2f}% ({v[0]}/{v[1]})")
        
    print("\n Phân tích theo Cấp học (Grade):")
    for k, v in metrics['grade'].items():
        if v[1] > 0: print(f"  - {k:<5} : {v[0]/v[1]*100:>5.2f}% ({v[0]}/{v[1]})")
    print("-" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    BASE_MODEL_PATH = os.path.join(BASE_DIR, "weights", "Qwen2-VL-2B-Instruct")
    QUANTIZED_MODEL_PATH = os.path.join(BASE_DIR, "weights", "Qwen2-VL-2B-Instruct-GPTQ-Int3")
    SFT_MODEL_PATH = os.path.join(BASE_DIR, "r3_quant_checkpoints")
    
    DATA_PATH = os.path.join(BASE_DIR, "data", "science_qa", "test-00000-of-00001-f0e719df791966ff.parquet")
    NUM_SAMPLES = 500
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        test_dataset = load_dataset("derek-thomas/ScienceQA", split="test")
        test_dataset.to_parquet(DATA_PATH)
        
    loader = ScienceQALocalLoader(DATA_PATH, subset_size=NUM_SAMPLES)
    df = loader.preprocess_for_r3_quant()

    base_acc, base_img_acc, base_metrics, base_preds = evaluate_model(BASE_MODEL_PATH, df)
    print_detailed_metrics("Base Model (16-bit)", base_acc, base_img_acc, base_metrics)

    quant_acc, quant_img_acc, quant_metrics, quant_preds = evaluate_model(QUANTIZED_MODEL_PATH, df)
    print_detailed_metrics("Quantized Model (3-bit)", quant_acc, quant_img_acc, quant_metrics)
    
    sft_acc, sft_img_acc, sft_metrics, sft_preds = evaluate_model(QUANTIZED_MODEL_PATH, df, lora_path=SFT_MODEL_PATH)
    print_detailed_metrics("FRPO Model (3-bit + LoRA)", sft_acc, sft_img_acc, sft_metrics)

    print("\n" + "="*60)
    print(f"BẢNG VÀNG THÀNH TÍCH ({NUM_SAMPLES} MẪU)")
    print("="*60)
    print(f"1. Base Model (16-bit)      : {base_acc:.2f}%")
    print(f"2. Quantized Model (3-bit)  : {quant_acc:.2f}%")
    print(f"3. GRPO Model (3-bit + LoRA): {sft_acc:.2f}%")
    print("-" * 60)
    print(f"GRPO đã phục hồi được       : {sft_acc - quant_acc:.2f}% so với bản 3-bit bị lỗi")
    print("="*60)