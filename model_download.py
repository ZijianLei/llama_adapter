# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
)
import torch
tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf",cache_dir='/home/comp/18482201/llm_research/zjlei/llama')
model = AutoModelForCausalLM.from_pretrained("yahma/llama-7b-hf",cache_dir='/home/comp/18482201/llm_research/zjlei/llama',
            
        )
model = model.to("cuda")
max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"max_allocated_memory: {max_allocated_memory/1024**2} MB")