
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29506 train_kd.py \
 --dataset_dir "./SemanticDataset_final" \
 --result_dir "./pths/DDRNet_KD_DINOv2" \
 --epochs 100 \
 --lr 1.e-2 \
 --batch_size 16 \
 --num_classes 19 \
 --crop_size 1024 1024 \
 --scale_range 0.75 1.25 \
 --teacher_type dinov2 \
 --teacher_arch dinov2_vits14 \
 --lambda_ce 1.0 \
 --lambda_kd_feat 1.0 \
 --save_every 20
