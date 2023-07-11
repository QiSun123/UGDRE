CUDA_VISIBLE_DEVICES=2 python -u  train.py --data_dir ./dataset/docred/ \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--dev_file dev_revised.json \
--test_file test_revised.json \
--train_batch_size 8 \
--test_batch_size 8 \
--gradient_accumulation_steps 4 \
--num_labels 5 \
--learning_rate 5e-5 \
--n_lr 1e-4 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 10.0 \
--seed 88 \
--num_class 97 \
--train_file train_revised.json \
--dropout 0.25 \
--resultPath ./results/result.json \
--save_path /data/sunqi/UGDRE/UGDRE-RE-finetune1 \
--finetune /data/sunqi/UGDRE/UGDRE-RE-pretrain1 \
--meta ./meta/rel2id.json