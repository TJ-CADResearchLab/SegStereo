model=acvnet
datapath=data
batch_size=2
logdir=acv_log
loadckpt=none
test_batch_size=1	
python3 main.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size
