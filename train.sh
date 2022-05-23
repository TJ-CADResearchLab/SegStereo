model=acvsgnet
datapath=data
batch_size=4
logdir=log/gwcsg_self_lr0.001_log
loadckpt=none
lr=0.001
test_batch_size=2	
python3 main.py --model $model --datapath $datapath --batch_size $batch_size \
--logdir $logdir --loadckpt $loadckpt --test_batch_size $test_batch_size --lr $lr --self_supervised 
