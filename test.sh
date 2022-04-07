OUTPUT_DIR=EV2_0_add_lane_gcn/test; \
GPU_NUM=1; \
CUDA_VISIBLE_DEVICES=0 python src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir train/data/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --sub_graph_batch_size 4096 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} --reuse_temp_file --temp_file_dir /research/dept7/zxzhao/jzwang/DenseTNT/output \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level point_level-4 \
    point_level-4-3 complete_traj complete_traj-3 \
  --model_recover_path EV2_0_add_lane_gcn/model_save/model.16.bin \
  --do_eval --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1 --eval_batch_size 8
