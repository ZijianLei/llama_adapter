export CUDA_VISIBLE_DEVICES=0,1,2,3
#train
PORT=$((20000 + RANDOM % 10000))
torchrun  --nproc_per_node=4 --master_port=$PORT finetune.py   \
    --base_model 'yahma/llama-7b-hf'   \
    --data_path 'ft-training_set/math_10k.json'  \
    --output_dir './trained_models/llama-7b-lora-math/'   \
    --batch_size 16  \
    --load_8bit \
    --micro_batch_size 4  \
    --num_epochs 3  \
    --learning_rate 3e-4  \
    --cutoff_len 256  \
    --val_set_size 0 \
    --eval_step 80 \
    --save_step 80 \
    --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 \
    --lora_alpha 64 
# exit n
# #inference
# torchrun generate.py \
#     --base_model 'yahma/llama-7b-hf' \
#     --lora_weights './trained_models/llama-7b-lora-math/' 
#evaluation
# for task in MultiArith #gsm8k AddSub AQuA SingleEq SVAMP
# do
#     torchrun --nproc_per_node=1 evaluate.py  --model LLaMA-7B --load_8bit  --adapter LoRA --dataset $task --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math/'     # specify the base model
#   #specify the adapter name ["LoRA", "AdapterH", "AdapterP", "Parallel"， "Scaled_Parallel""]
#          #specify the test dataset      
# done