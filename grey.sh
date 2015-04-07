export THEANO_FLAGS=cuda.root=/usr/local/cuda,device=gpu

# Basic
python train_multi.py --data_augmentation --rotate --crop --flip \
                      --lr 0.005 --momentum_factor 0.9 --NAG \
                      --arch 3 --height 112 --width 112 \
                      --nhu0 64 --kw0 9 --pool0 2 \
                      --nhu1 128 --kw1 9 --pool1 2 \
                      --nhu2 256 --kw2 7 --pool2 2 \
                      --nhu3 512 --kw3 7 --pool3 2 \
                      --nhu4 512 \
                      --xp_path xp/grey_color_bigger \

