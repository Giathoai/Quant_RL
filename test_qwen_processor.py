import torch
from transformers import AutoProcessor
from PIL import Image

def test():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    # Create a dummy image
    image = Image.new('RGB', (224, 224), color = 'red')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is in this image?"}
            ]
        }
    ]
    
    print("Testing apply_chat_template...")
    try:
        inputs = processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
