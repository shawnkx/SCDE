#sleep 3h
TASK_NAME=SENCLZ
GLUE_DIR=$1
# for lr in 1e-5 2e-5 3e-5
# do
# CUDA_VISIBLE_DEVICES=0,1 python ./examples/run_glue.py --model_type bert --model_name_or_path bert-large-uncased --task_name SENCLZ  --do_lower_case --data_dir ${GLUE_DIR} --max_seq_length 256 --learning_rate ${lr} --num_train_epochs 2.0 --output_dir ../final_data_exp/models/bert_spn_large_models_epoch2_${lr} --do_eval --gradient_accumulation_steps 2  --per_gpu_train_batch_size 8 --do_train
# done

for lr in 2e-5
do
CUDA_VISIBLE_DEVICES=0,1 python ./examples/run_glue.py --model_type bert --model_name_or_path bert-large-uncased --task_name SENCLZ  --do_lower_case --data_dir ${GLUE_DIR} --max_seq_length 256 --learning_rate ${lr} --num_train_epochs 3.0 --output_dir ../final_data_exp/models/apn_models/bert_apn_large_models_epoch3_${lr} --do_eval --gradient_accumulation_steps 1  --per_gpu_train_batch_size 16
done


# CUDA_VISIBLE_DEVICES=0 python3 examples/run_head_to_span.py --data_dir ../event/ --model_type bert --model_name_or_path bert-base-cased --output_dir ../bert_hts_2e-5 --max_seq_length 128 --num_train_epochs 3 --per_gpu_train_batch_size 32  --save_steps 0 --seed 42  --do_eval --learning_rate 1e-5  --do_train --output_dir ../event/bert_hts_1e-5 --num_train_epochs 3
# CUDA_VISIBLE_DEVICES=0 python3 examples/run_head_to_span.py --data_dir ../event/ --model_type bert --model_name_or_path bert-base-cased --output_dir ../bert_hts_2e-5 --max_seq_length 128 --num_train_epochs 3 --per_gpu_train_batch_size 32  --save_steps 0 --seed 42  --do_eval --learning_rate 3e-5  --do_train --output_dir ../event/bert_hts_3e-5 --num_train_epochs 3
# CUDA_VISIBLE_DEVICES=0 python3 examples/run_head_to_span.py --data_dir ../event/ --model_type bert --model_name_or_path bert-base-cased --output_dir ../bert_hts_2e-5 --max_seq_length 128 --num_train_epochs 3 --per_gpu_train_batch_size 32  --save_steps 0 --seed 42  --do_eval --learning_rate 2e-5  --do_train --output_dir ../event/bert_hts_2e-5 --num_train_epochs 3
