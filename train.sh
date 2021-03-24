python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --channel_multiplier 1 --batch 4 --size 256 --lr 0.002 --iter 3800000 lmdb_data_path
