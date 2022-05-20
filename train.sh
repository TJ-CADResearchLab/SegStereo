model=acvsgnet
datapath=data
batch_size=2
logdir=log/acvsg_refine_onlyrefine_log
loadckpt=log/acvsg_log/checkpoint_000_epe_2.011.ckpt
lr=0.001
test_batch_size=2	
python3 main.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size --lr $lr --refine_mode --only_train_refine 
