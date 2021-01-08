cd /home/zsy/jupyter/IR-Net/CIFAR-10/VGG-Small


#CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/bin/python3 ./fullbit_vgg.py > ./out/fullbit1.out &
#CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./fullbit_vgg.py > ./out/fullbit2.out &
#CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py ./out/allnormal1/ > ./out/allnormal1.out &
#CUDA_VISIBLE_DEVICES=1 /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py ./out/allnormal2/ > ./out/allnormal2.out &

#CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal1/ > ./out/partnormal1.out &
#CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal2/ > ./out/partnormal2.out &
#CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal3/ > ./out/partnormal3.out &
#CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./IR-Net_firstnormal_vgg.py ./out/partnormal4/ > ./out/partnormal4.out &

CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=32 -save_dir=./out/allnormal32_1/ > ./out/allnormal32_1.out &
CUDA_VISIBLE_DEVICES=0 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=32 -save_dir=./out/allnormal32_2/ > ./out/allnormal32_2.out &

CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=16 -save_dir=./out/allnormal16_1/ > ./out/allnormal16_1.out &
CUDA_VISIBLE_DEVICES=1 nohup /home/ubuntu/miniconda3/bin/python3 ./all_normal_vgg.py -quan_bit=16 -save_dir=./out/allnormal16_2/ > ./out/allnormal16_2.out &

exit