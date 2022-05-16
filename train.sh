model=acvsgnet
datapath=data
batch_size=8
logdir=log/acvsg_self_occ_refine_log
loadckpt=log/acvsg_self_occ_log/checkpoint_004_epe_5.541.ckpt
lr=0.0001
test_batch_size=1	
python3 main.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size --lr $lr  --train_seg
