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

python3.7 env_gen/generate_warehouse_v1_mazes.py --seed ${1:-1} --dir environments/buckets_five_five_100_20 --train_bucket_to_boxes B b C c D d E e F f --test_bucket_to_boxes G g H h I i J j K k --num_train 100 --num_test 20
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c D d E e F f G g H h I i J j K k --output_file environments/buckets_five_five_100_20/kg.json
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c D d E e F f G g H h I i J j K k --output_file environments/buckets_five_five_100_20/kg_colors.json --use_colors
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c D d E e F f G g H h I i J j K k --output_file environments/buckets_five_five_100_20/kg_same_edge_feats.json --same_edge_feats
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c D d E e F f G g H h I i J j K k --output_file environments/buckets_five_five_100_20/kg_no_edges.json --no_edges
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c D d E e F f G g H h I i J j K k --output_file environments/buckets_five_five_100_20/kg_fc.json --fully_connected
python3.7 env_gen/generate_warehouse_v1_kg.py --bucket_to_boxes B b C c D d E e F f G g H h I i J j K k --output_file environments/buckets_five_five_100_20/kg_fc_distinct.json --fully_connected_distinct