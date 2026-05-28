#!/bin/bash
set -e

for manifest in manifests/*.txt
do
    name=$(basename "$manifest" .txt)

    DOWNLOAD_DIR="./downloads/$name"
    RESULT_DIR="./results/$name"
    PROCESS_CSV="./process_lists/$name.csv"

    mkdir -p "$DOWNLOAD_DIR" "$RESULT_DIR"

    echo "===== DOWNLOAD ====="
    gdc-client download \
        -m "$manifest" \
        -d "$DOWNLOAD_DIR"

    echo "===== PATCH ====="
    python create_patches_fp.py \
        --source "$DOWNLOAD_DIR" \
        --save_dir "$RESULT_DIR" \
        --process_list "$PROCESS_CSV"

    echo "===== UNI FEATURE ====="
    CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
        --data_h5_dir "$RESULT_DIR/patches" \
        --data_slide_dir "$DOWNLOAD_DIR" \
        --csv_path "$PROCESS_CSV" \
        --feat_dir "$RESULT_DIR/features" \
        --model_name uni

    echo "===== COMPRESS FEATURE H5 ====="
    tar -czf "$RESULT_DIR/features/h5_files.tar.gz" \
        -C "$RESULT_DIR/features" h5_files

    echo "===== CHECK PT FEATURES ====="
    ls "$RESULT_DIR/features/pt_files"/*.pt >/dev/null

    echo "===== DELETE RAW SVS ====="
    rm -rf "$DOWNLOAD_DIR"

    echo "===== DELETE UNCOMPRESSED FEATURE H5 ====="
    rm -rf "$RESULT_DIR/features/h5_files"

    echo "===== DELETE ORIGINAL H5 ====="
    rm -rf "$RESULT_DIR/patches"

    echo "===== DONE $name ====="
done

