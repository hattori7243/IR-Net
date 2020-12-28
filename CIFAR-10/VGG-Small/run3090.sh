cd /home/zsy/jupyter/IR-Net/CIFAR-10/VGG-Small
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/bin/python3 ./IR-Net_origin.py > ./out/origin1.out &
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/bin/python3 ./IR-Net_quan_normal.py ./out/normal1/ > ./out/quan_normal1.out &
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./IR-Net_origin.py > ./out/origin2.out &
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./IR-Net_quan_normal.py ./out/normal2/ > ./out/quan_normal2.out &