export HCCL_IF_IP=xx.xx.xx.xx
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
export DISAGGREGATED_RPEFILL_RANK_TABLE_PATH="/home/xxx/ranktable_118.json"
export ASCEND_RT_VISIBLE_DEVICES=3
#export VLLM_LLMDD_CHANNEL_PORT=6658
#export VLLM_LLMDD_CHANNEL_PORT=15272
export VLLM_LLMDD_CHANNEL_PORT=14012
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCAKE_CONFIG_PATH="/home/xxx/vllm/mooncake_barebone.json"
export VLLM_USE_V1=1
export VLLM_BASE_PORT=9800
export ENV_RANKTABLE_PATH="/home/xxx/hccl_8p_01234567_xx.xx.xx.xx.json"

vllm serve "/home/z00841663/ckpt/model/Qwen2.5-7B-Instruct" \
  --host xx.xx.xx.xx \
  --port 8200 \
  --tensor-parallel-size 1\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000  \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1_barebone",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20002",
  "engine_id": 1,
  "kv_rank": 1,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector_v1_barebone",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 1
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 1
             }
      }
  }'  \
  --additional-config \
  '{}'\

