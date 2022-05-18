model=acvsgnet
datapath=data
batch_size=6
logdir=valid_log
loadckpt=log/acvsg_conbine_refine_log/checkpoint_000_epe_2.049_42.047.ckpt
test_batch_size=1	
python3 valid.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size
