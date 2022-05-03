model=ednet
datapath=data
batch_size=6
logdir=valid_log
loadckpt=ed_log/checkpoint_000_epe_4.276.ckpt
test_batch_size=1	
python3 valid.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size
