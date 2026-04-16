import sys
import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, GPTQConfig, AutoConfig

_original_init = Qwen2VLForConditionalGeneration.__init__
def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.hf_device_map = {"": "cuda:0"}  # Khởi tạo là có ngay bản đồ thiết bị!
Qwen2VLForConditionalGeneration.__init__ = _patched_init
# ==============================================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

class QwenGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def get_calibration_data(self, test_size=8):
        loader = ScienceQALocalLoader(self.data_path, subset_size=test_size)
        df = loader.preprocess_for_r3_quant()
        return [f"Question: {row['question']}\nAnswer: {row['reasoning']}" for _, row in df.iterrows()]

    def quantize_and_save(self, bits=3):
        calib_dataset = self.get_calibration_data(test_size=8)
        
        gptq_config = GPTQConfig(
            bits=bits,
            dataset=calib_dataset,
            tokenizer=self.base_model_path, 
            use_exllama=False,            
            desc_act=False,
            sym=True
        )

        config = AutoConfig.from_pretrained(self.base_model_path)
        config.use_cache = False

        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config,
                quantization_config=gptq_config,
                device_map={"": "cuda:0"},  # Ép cứng dict map thay vì "auto"
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            
            print("\n" + "="*50)
            print("🎉 LƯỢNG TỬ HÓA VÀ ĐÓNG GÓI THÀNH CÔNG RỰC RỠ!")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"--- error: {e} ---")
            sys.exit(1)