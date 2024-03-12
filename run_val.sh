#!/bin/sh

test_filename="gt_val.json"
results_dir="./val_results"

mkdir "$results_dir/metrics"
for filename in "$results_dir/"*.json; do
    if [ "$filename" == "$results_dir""$test_filename" ] ; then
        continue;
    fi
    echo "processing $filename with GT $results_dir/$test_filename and saving $results_dir/metrics/$(basename ${filename})"
    python eval.py --pred-path "$filename" --gt-path "$results_dir/$test_filename"  --save-path "$results_dir/metrics/$(basename ${filename})"
done