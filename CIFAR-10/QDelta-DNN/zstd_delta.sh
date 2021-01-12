#!/bin/bash
int=0
src_dir_path=./all_delta2
dst_dir_path=./delta_zstd
if [ -d $dst_dir_path ]
then
  rm -rf $dst_dir_path
fi
mkdir $dst_dir_path
while (($int <=998))
do
   echo "$src_dir_path/$int.out compress to $dst_dir_path/$int.zstd"
   zstd -o $dst_dir_path/$int.zstd -10 $src_dir_path/$int.out
   let "int++"
done
du -lh|grep $dst_dir_path
exit