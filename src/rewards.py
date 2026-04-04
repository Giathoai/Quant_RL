import re

def extract_xml_answer(text: str) -> str:
    """Trích xuất nội dung nằm trong thẻ <answer>...</answer>"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Format Reward (Chuẩn Open-R1): Chỉ thưởng 1.0 nếu định dạng đúng tuyệt đối.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        if re.match(pattern, text.strip(), re.DOTALL | re.IGNORECASE):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Accuracy Reward (Chuẩn Open-R1): Chỉ thưởng 1.0 nếu trùng khớp chính xác tuyệt đối.
    """
    index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'} 
    rewards = []
    
    for comp, truth in zip(completions, ground_truth):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        pred_answer = extract_xml_answer(text)
        
        if isinstance(truth, int) or (isinstance(truth, str) and truth.isdigit()):
            truth_clean = index_to_letter.get(int(truth), str(truth)).lower().strip()
        else:
            truth_clean = str(truth).lower().strip()
            
        pred_clean = pred_answer.lower().strip()
        
        if pred_clean == truth_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards