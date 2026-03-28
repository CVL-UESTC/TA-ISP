export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=4,5 python main.py \
    -f cfg_small \
    -d 2 \
    -b 16 \
    -expn experiment_paper/RAOD_taisp36_tiny_day