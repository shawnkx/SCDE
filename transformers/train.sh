TASK_NAME=SENCLZ
GLUE_DIR=$1
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 python ./examples/run_glue.py --model_type bert --model_name_or_path bert-large-uncased --task_name SENCLZ  --do_lower_case --data_dir ${GLUE_DIR} --max_seq_length 256 --learning_rate ${lr} --num_train_epochs 3.0 --output_dir models/apn_models/bert_apn_large_models_${lr} --do_eval --gradient_accumulation_steps 1  --per_gpu_train_batch_size 16 --do_train


