CUDA_VISIBLE_DEVICES=0 python train.py \
        --batch_size 196 \
        --epochs 100 \
        --model mae_vit_base_patch16 \
        --input_size 224 \
        --weight_decay 0.05 \
        --blr 1e-3 \
        --warmup_epochs 5 \
        --dataset_name_train "inpaint-context/train-mae-update-furniture" \
        --image_folder \
            "/mnt/Datadrive/tiennv/data/final" \
            "/mnt/Datadrive/datasets/ade20k/ade20k" \
            "/mnt/Datadrive/datasets/ade20k/pascal-context" \
            "/mnt/Datadrive/datasets/coco2017/train" \
            "/mnt/Datadrive/datasets/coco2017/val" \
        --do_train \
        --do_eval \
        --mask_ratio 0.5 \
        --mask_min 0 \
        --mask_max 1 \
        --cache_dir .cache \
        --output_dir outputs/files \
        --log_dir outputs/logs \
        --weights checkpoints/mae_visualize_vit_base.pth \
        --mask_mode 'objmask' 
