# HoliTom
WRAPPER=holitom RETAIN_RATIO=0.10 T=0.65 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.10 2>&1 | tee ./logs/ov-7b-holitom/0.10/ov-7b-holitom-0.10-t0.65-k18-r0.5.log
WRAPPER=holitom RETAIN_RATIO=0.15 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.15 2>&1 | tee ./logs/ov-7b-holitom/0.15/ov-7b-holitom-0.15-t0.80-k18-r0.5.log
WRAPPER=holitom RETAIN_RATIO=0.20 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.20 2>&1 | tee ./logs/ov-7b-holitom/0.20/ov-7b-holitom-0.20-t0.80-k18-r0.5.log
WRAPPER=holitom RETAIN_RATIO=0.25 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.25 2>&1 | tee ./logs/ov-7b-holitom/0.25/ov-7b-holitom-0.25-t0.80-k18-r0.5.log

# HoliTom (w/o M)
WRAPPER=holitom RETAIN_RATIO=0.10 T=0.65 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.10 2>&1 | tee ./logs/ov-7b-holitom/0.10/ov-7b-holitom-0.10-t0.65.log
WRAPPER=holitom RETAIN_RATIO=0.15 T=0.80 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.15 2>&1 | tee ./logs/ov-7b-holitom/0.15/ov-7b-holitom-0.15-t0.80.log
WRAPPER=holitom RETAIN_RATIO=0.20 T=0.80 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.20 2>&1 | tee ./logs/ov-7b-holitom/0.20/ov-7b-holitom-0.20-t0.80.log
WRAPPER=holitom RETAIN_RATIO=0.25 T=0.80 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks mvbench,egoschema,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-holitom/0.25 2>&1 | tee ./logs/ov-7b-holitom/0.25/ov-7b-holitom-0.25-t0.80.log
