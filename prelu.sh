export THEANO_FLAGS=cuda.root=/usr/local/cuda,device=gpu

#python train_multi.py --data_augmentation --rotate --crop --flip \
#                      --lr 0.005 --momentum_factor 0.9 --NAG \
#                      --L1_factor 0.0001 --L2_factor 0.0001 \
#                      --arch 3 --height 97 --width 97 \
#                      --nhu0 32 --kw0 8 --pool0 2 \
#                      --nhu1 64 --kw1 8 --pool1 2 \
#                      --nhu2 128 --kw2 6 --pool2 2 \
#                      --nhu3 256 --kw3 6 --pool3 2 \
#                      --nhu4 256 \
#                      --xp_path xp/prelu12_big \
#
python train_multi.py --data_augmentation --rotate --crop --flip \
                      --lr 0.005 --momentum_factor 0.9 --NAG \
                      --L1_factor 0.00005 --L2_factor 0.00005 \
                      --arch 8 --height 82 --width 82 \
                      --nhu0 32 --kw0 4 --pool0 2 \
                      --nhu1 64 --kw1 4 --pool1 2 \
                      --nhu2 128 --kw2 3 --pool2 2 \
                      --nhu3 256 --kw3 3 --pool3 2 \
                      --nhu4 256 \
                      --xp_path xp/test_deeper3 \
                      



#python train_multi.py --data_augmentation --rotate --crop --flip --grey \
#                      --lr 0.005 --momentum_factor 0.9 --NAG \
#                      --L1_factor 0.0001 --L2_factor 0.0001 \
#                      --arch 3 --height 97 --width 97 \
#                      --nhu0 32 --kw0 8 --pool0 2 \
#                      --nhu1 64 --kw1 8 --pool1 2 \
#                      --nhu2 128 --kw2 6 --pool2 2 \
#                      --nhu3 256 --kw3 6 --pool3 2 \
#                      --nhu4 256 \
#                      --xp_path xp/prelu12_big_grey \
#
#python train_multi.py --data_augmentation --rotate --crop --flip --grey \
#                      --lr 0.005 --momentum_factor 0.9 --NAG \
#                      --L1_factor 0.0001 --L2_factor 0.0001 \
#                      --arch 5 --height 49 --width 49 \
#                      --nhu0 32 --kw0 4 --pool0 2 \
#                      --nhu1 64 --kw1 4 --pool1 2 \
#                      --nhu2 128 --kw2 3 --pool2 2 \
#                      --nhu3 256 --kw3 3 --pool3 2 \
#                      --nhu4 256 \
#                      --xp_path xp/prelu12_small_grey \
