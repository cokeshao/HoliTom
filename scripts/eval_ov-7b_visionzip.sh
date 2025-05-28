# 0.10
WRAPPER=visionzip SPATIAL_TOKENS=20 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-visionzip/0.10 2>&1 | tee ./logs/ov-7b-visionzip/0.10/ov-7b-visionzip-0.10.log

# 0.15
WRAPPER=visionzip SPATIAL_TOKENS=30 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-visionzip/0.15 2>&1 | tee ./logs/ov-7b-visionzip/0.15/ov-7b-visionzip-0.15.log

# 0.20
WRAPPER=visionzip SPATIAL_TOKENS=40 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-visionzip/0.20 2>&1 | tee ./logs/ov-7b-visionzip/0.20/ov-7b-visionzip-0.20.log

# 0.25
WRAPPER=visionzip SPATIAL_TOKENS=50 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-visionzip/0.25 2>&1 | tee ./logs/ov-7b-visionzip/0.25/ov-7b-visionzip-0.25.log
