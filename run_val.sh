#!/bin/sh
set -e

test_filename="gt_test.json"
results_dir="./test_results"

mkdir -p "$results_dir"

# generate results
#for filename in "./configs/models/"*.yaml; do
#    echo "generating test set results for $filename"
#    python train.py --config-file "$filename" --test
#done

# generating metrics
mkdir -p "$results_dir/metrics"
mkdir -p "$results_dir/metrics/1"
mkdir -p "$results_dir/metrics/2"
mkdir -p "$results_dir/metrics/3"

for filename in "$results_dir/"*.json; do
    if [ $(basename ${filename}) == "$test_filename" ] ; then
        continue;
    fi
    echo "processing $filename with GT $results_dir/$test_filename and saving $results_dir/metrics/$(basename ${filename})"
    python eval.py --pred-path "$filename" --gt-path "$results_dir/$test_filename"  --save-path "$results_dir/metrics/$(basename ${filename})"
    python eval.py --pred-path "$filename" --gt-path "$results_dir/$test_filename"  --save-path "$results_dir/metrics/1/$(basename ${filename})" --num-segments 1
    python eval.py --pred-path "$filename" --gt-path "$results_dir/$test_filename"  --save-path "$results_dir/metrics/2/$(basename ${filename})" --num-segments 2
    python eval.py --pred-path "$filename" --gt-path "$results_dir/$test_filename"  --save-path "$results_dir/metrics/3/$(basename ${filename})"  --num-segments 3
done