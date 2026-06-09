#!/bin/bash
set -e

for manifest in manifests/*.txt
do
    name=$(basename "$manifest" .txt)

    DOWNLOAD_DIR="/home/yuz/pathML/data/$name"
    WSI_DIR="/home/yuz/pathML/wsi_files/$name"
    RESULT_DIR="/home/yuz/pathML/patches/$name"
    PROCESS_CSV="/home/yuz/pathML/process_lists/$name.csv"

    mkdir -p "$DOWNLOAD_DIR" "$WSI_DIR" "$RESULT_DIR" "$(dirname "$PROCESS_CSV")"

    echo "===== DOWNLOAD ====="
    gdc-client download \
        -m "$manifest" \
        -d "$DOWNLOAD_DIR"

    echo "===== MOVE SVS FILES ====="
    find "$DOWNLOAD_DIR" -name "*.svs" -print0 | xargs -0 -r mv -t "$WSI_DIR"

    echo "===== PATCH ====="
    python create_patches_fp.py \
        --source "$WSI_DIR" \
        --save_dir "$RESULT_DIR" \
        --process_list "$PROCESS_CSV"

    echo "===== UNI FEATURE ====="
    CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
        --data_h5_dir "$RESULT_DIR/patches" \
        --data_slide_dir "$WSI_DIR" \
        --csv_path "$PROCESS_CSV" \
        --feat_dir "$RESULT_DIR/features" \
        --model_name uni

    echo "===== COMPRESS FEATURE H5 ====="
    tar -czf "$RESULT_DIR/features/h5_files.tar.gz" \
        -C "$RESULT_DIR/features" h5_files

    echo "===== CHECK PT FEATURES ====="
    ls "$RESULT_DIR/features/pt_files"/*.pt >/dev/null

    echo "===== DELETE RAW SVS ====="
    rm -rf "$WSI_DIR"

    echo "===== DELETE UNCOMPRESSED FEATURE H5 ====="
    rm -rf "$RESULT_DIR/features/h5_files"

    echo "===== DELETE ORIGINAL H5 ====="
    rm -rf "$RESULT_DIR/patches"
    rm -rf "$DOWNLOAD_DIR"

    echo "===== COMPRESS PT FEATURES ====="
    tar -czf "$RESULT_DIR/features/pt_files.tar.gz" \
        -C "$RESULT_DIR/features" pt_files

    echo "===== VERIFY PT TAR ====="
    tar -tzf "$RESULT_DIR/features/pt_files.tar.gz" >/dev/null

    echo "===== DONE $name ====="
    
done