
WRAPPER=holitom RETAIN_RATIO=0.15 T=0.80 FASTV_k=60 FASTV_r=0.5 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=1 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-72b-ov-sft,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32,device_map=auto \
--tasks mvbench,egoschema,videomme,longvideobench_val_v  \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-72b-holitom/0.15 2>&1 | tee ./logs/ov-72b-holitom/0.15/ov-72b-holitom-0.15-t0.80-k60-r0.5.log
WRAPPER=holitom RETAIN_RATIO=0.15 T=0.80 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=1 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-72b-ov-sft,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32,device_map=auto \
--tasks mvbench,egoschema,videomme,longvideobench_val_v  \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-72b-holitom/0.15 2>&1 | tee ./logs/ov-72b-holitom/0.15/ov-72b-holitom-0.15-t0.80.log

