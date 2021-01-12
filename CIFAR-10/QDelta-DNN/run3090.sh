cd /home/zsy/jupyter/IR-Net/CIFAR-10/QDelta-DNN



CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_quan_vgg.py -quan_bit=8 -save_dir=./train_out/all_normal_quan/8bit_1/ > ./train_out/all_normal_quan/8bit_1.out &

CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_quan_vgg.py -quan_bit=8 -save_dir=./train_out/all_quan/8bit_1/ > ./train_out/all_quan/8bit_1.out &

CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./fullbit_vgg.py -save_dir=./train_out/full_bit/1/ > ./train_out/full_bit/1.out &

CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py -save_dir=./train_out/first_normal_quan/1/ > ./train_out/first_normal_quan/1.out &



exit