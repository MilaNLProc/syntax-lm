export TASK_NAME=cola
export COMET_API_KEY=n8jLPmZqhs7Xs96UlkTGdIk5p

python train.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir /data1/attanasiog/syntax/roberta/$TASK_NAME/ \
  --overwrite_output_dir \
  --report_to comet_ml \
  --warmup_ratio 0.1 \
  --lr_scheduler_type linear \
  --load_best_model_at_end \
  --save_total_limit 2 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --logging_steps 50 \
  --save_steps 50 \
  --save_strategy steps \
  --dataloader_num_workers 4
