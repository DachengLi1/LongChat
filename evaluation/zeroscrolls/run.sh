#scrolls (200 validation samples)
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_32K_interpolate" --flash --dataset qasper --ratio 16
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_7b_16K" --flash --dataset qasper
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_13b_16K" --flash --dataset qasper
 
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_32K_interpolate" --flash --dataset narrative_qa --ratio 16
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_7b_16K" --flash --dataset narrative_qa
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_13b_16K" --flash --dataset narrative_qa

python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-7b-v1.3" --flash --dataset qasper
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-7b-v1.3" --flash --dataset narrative_qa

python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-13b-v1.3" --flash --dataset qasper
python eval.py --dataset-version "tau/scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-13b-v1.3" --flash --dataset narrative_qa
 

# zero scrolls
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_32K_interpolate" --flash --dataset qasper
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_7b_16K" --flash --dataset qasper
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_13b_16K" --flash --dataset qasper
#  
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_32K_interpolate" --flash --dataset narrative_qa
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_7b_16K" --flash --dataset narrative_qa
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/dacheng-data/longchat_13b_16K" --flash --dataset narrative_qa
# 
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-7b-v1.3" --flash --dataset qasper
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-7b-v1.3" --flash --dataset narrative_qa
 
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-13b-v1.3" --flash --dataset qasper
# python eval.py --dataset-version "tau/zero_scrolls" --model-name-or-path "/home/haozhang/LongChat/data/vicuna-13b-v1.3" --flash --dataset narrative_qa
 
