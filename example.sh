SAVE_NAME="distribution.png"
START_LEVEL=-10
END_LEVEL=2
FOLDERNAME_LIST="/mnt/nvme0n1p1/tmp/parameters/0 /mnt/nvme0n1p1/tmp/parameters/10"

python example.py \
    --foldername_list $FOLDERNAME_LIST \
    --save_name $SAVE_NAME \
    --start_level $START_LEVEL \
    --end_level $END_LEVEL \
    --verbose
