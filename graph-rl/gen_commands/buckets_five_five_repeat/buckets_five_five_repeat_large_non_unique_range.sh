# ******************************************************************************
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

python3.7 env_gen/generate_warehouse_v1_mazes.py --seed ${1:-1} --dir environments/buckets_five_five_repeat_large_non_unique_range --train_bucket_to_boxes "66" "0" "3" "4" "5" "6" "7" "8" "14" "15" "67" "16" "17" "18" "19" "20" "21" "22" "23" "24" "68" "25" "26" "27" "33" "34" "35" "36" "37" "38" "69" "39" "40" "41" "42" "44" "45" "46" "47" "48" "70" "49" "50" "51" "52" "53" "54" "55" "56" "57"  --test_bucket_to_boxes "66" "0" "3" "4" "5" "6" "7" "8" "14" "15" "67" "16" "17" "18" "19" "20" "21" "22" "23" "24" "68" "25" "26" "27" "33" "34" "35" "36" "37" "38" "69" "39" "40" "41" "42" "44" "45" "46" "47" "48" "70" "49" "50" "51" "52" "53" "54" "55" "56" "57" --use_raw_ascii --num_train 100 --num_test 100  --min_num_buckets 1 --max_num_buckets 3
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b c d C e f g D h i j E k l m F n o p --output_file environments/buckets_five_five_repeat_large_non_unique_range/kg.json
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b c d C e f g D h i j E k l m F n o p --output_file environments/buckets_five_five_repeat_large_non_unique_range/kg_colors.json --use_colors