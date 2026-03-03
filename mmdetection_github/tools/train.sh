CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=25000 bash tools/dist_train.sh \
    configs/LOD/R_Net_taisp.py 4 \
    --work-dir work_dirs/LOD_R_Net_taisp

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=25000 bash tools/dist_train.sh \
    configs/PASCALRAW_Res18/Normal_Light_taisp_res18.py 4 \
    --work-dir work_dirs/PASCALRAW_taisp_res18

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=25000 bash tools/dist_train.sh \
    configs/PASCALRAW_Res50/Normal_Light_taisp_res50.py 4 \
    --work-dir work_dirs/PASCALRAW_taisp_res50






