OUTPUT_DIR=set_EV2_0_add_lane_gcn; \
MODEL_PATH=EV2_0_add_lane_gcn/model_save/model.16.bin; \
GPU_NUM=4; \
python src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir train/data/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} --reuse_temp_file --temp_file_dir /research/dept7/zxzhao/jzwang/DenseTNT/output \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    set_predict-train_recover=${MODEL_PATH} \
