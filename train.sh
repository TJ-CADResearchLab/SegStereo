model=attnet
datapath=data
batch_size=4
logdir=att_log
loadckpt=none
python3 main.py --model $model --datapath $datapath --batch_size $batch_size --logdir $logdir --loadckpt $loadckpt
