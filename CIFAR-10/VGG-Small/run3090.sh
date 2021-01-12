cd /home/zsy/jupyter/IR-Net/CIFAR-10/VGG-Small


#CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py ./out/allnormal1/ > ./out/allnormal1.out &
#CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py ./out/allnormal2/ > ./out/allnormal2.out &

#CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal1/ > ./out/partnormal1.out &
#CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal2/ > ./out/partnormal2.out &
#CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal3/ > ./out/partnormal3.out &
#CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal4/ > ./out/partnormal4.out &

CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./fullbit_vgg.py -save_dir=./train_out/full_bit/1 > ./train_out/full_bit/1.out &
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./fullbit_vgg.py -save_dir=./train_out/full_bit/2 > ./train_out/full_bit/2.out &

CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=8 -save_dir=./train_out/all_normal/8bit_1/ > ./train_out/all_normal/8bit_1.out &
CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=8 -save_dir=./train_out/all_normal/8bit_2/ > ./train_out/all_normal/8bit_2.out &


#CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=32 -save_dir=./train_out/all_normal/32bit_2/ > ./train_out/all_normal/32bit_2.out &
#CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=32 -save_dir=./train_out/all_normal/32bit_3/ > ./train_out/all_normal/32bit_3.out &

#CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=16 -save_dir=./train_out/all_normal/16bit_3/ > ./train_out/all_normal/16bit_3.out &
#CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=16 -save_dir=./train_out/all_normal/16bit_4/ > ./train_out/all_normal/16bit_4.out &

exit