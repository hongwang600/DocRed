export CUDA_VISIBLE_DEVICES=1
MODEL_NAME=sDNC # LSTM, BiLSTM, BiDNC, sDNC
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')

python3 train.py \
    --model_name $MODEL_NAME \
    --save_name ${MODEL_NAME}_${NOW} \
    --train_prefix dev_train \
    --test_prefix dev_dev \
