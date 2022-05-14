model=acvsgnet
datapath=data
batch_size=6
logdir=valid_log
loadckpt=log/acvsg_self_log/checkpoint_009_epe_6.339.ckpt
test_batch_size=1	
python3 valid.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size
