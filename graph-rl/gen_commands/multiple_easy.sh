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

python3.7 env_gen/generate_warehouse_v1_mazes.py --seed ${1:-1} --dir environments/multiple_easy_100_20 --train_bucket_to_boxes B b C c --test_bucket_to_boxes B b C c --num_train 100 --num_test 20 --min_num_buckets 2 --max_num_buckets 2
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c --output_file environments/multiple_easy_100_20/kg.json
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c --output_file environments/multiple_easy_100_20/kg_colors.json --use_colors
