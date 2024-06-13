SAVE_NAME="gpus_2_step22_fp16_variance(scaled).png"
START_LEVEL=-10
END_LEVEL=5
FOLDERNAME_LIST="/mnt/nvme0n1p1/dumper_update/step_22/rank_0/dumptype_variance /mnt/nvme0n1p1/dumper_update/step_22/rank_1/dumptype_variance"
FILENAME_CONSTRAINT="float16"

python example.py \
    --foldername_list $FOLDERNAME_LIST \
    --save_name $SAVE_NAME \
    --start_level $START_LEVEL \
    --end_level $END_LEVEL \
    --filename_constraint $FILENAME_CONSTRAINT \
    --verbose