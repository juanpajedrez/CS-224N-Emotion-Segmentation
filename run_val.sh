#!/bin/sh

test_filename="gt_test.json"
results_dir="./test_results"
mkdir "$results_dir"

# generate results
for filename in "./configs/models/"*.json; do
    echo "generating test set results for $filename"
    python --config-file "$filename" --test
done

# generating metrics
mkdir "$results_dir/metrics"
for filename in "$results_dir/"*.json; do
    if [ "$filename" == "$results_dir""$test_filename" ] ; then
        continue;
    fi
    echo "processing $filename with GT $results_dir/$test_filename and saving $results_dir/metrics/$(basename ${filename})"
    python eval.py --pred-path "$filename" --gt-path "$results_dir/$test_filename"  --save-path "$results_dir/metrics/$(basename ${filename})"
done