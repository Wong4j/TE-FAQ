![image](https://github.com/Wong4j/TE-FAQ/assets/21985950/5522b29c-fab5-45cf-bfde-e9f34525cd7d)# TE-FAQ
收集Transformer Engine(TE)测试常见的问题和注意事项


## Table Of Contents

- [NeMo](#nemo)
    * [环境](#environment)
    * [数据集](#dataset)
    * [训练脚本和配置](#config)
- [PaddleNLP](#paddlenlp)
    * [训练脚本和配置](#scripts-and-config)
    * [收敛性测试](#convergence)
- [Advanced](#advanced)


## NeMo
NeMo的测试方法先参考https://github.com/Wong4j/nemo_test/blob/main/README.md 和官方的example https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/examples/README.md


### 环境
Q: NeMo测试用什么镜像？   
A: 最好直接用NeMo的镜像，在https://developer.nvidia.com/nemo-framework/join 上申请NeMo Framework Container的GA。用尽量新的image，比如nvcr.io/nvidia/nemo:24.03.framework。
官方镜像基于NGC PyTorch的镜像，预装了TE, NeMo, NeMo-Framework-Launcher等。

Q: 如果官方镜像不适用怎么办？比如想用Ubuntu 20.04   
A: 建议参考官方的[Dockerfile](https://github.com/NVIDIA/NeMo/blob/main/Dockerfile)修改，然后自己build镜像


### 数据集
Q: Benchmark用什么数据集？   
A: 如果不做收敛性测试，建议直接用synthetic dataset，方法是在yaml文件中修改
`model.data.data_impl="mock"`和`model.data.data_prefix=[]`

### 训练脚本和配置
Q: 单机interactive模式怎么跑？   
A: 除了官方的example中给的方法，也可以直接跑`CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u /path/to/megatron_gpt_pretraining.py  --config-path=xxx --config-name=xxx`

Q: LLaMA和GPT的脚本是一样的吗？   
A: 是的，实际训练都是通过megatron_gpt_pretraining.py启动，仅yaml配置不同。

Q: 为什么会OOM？   
A: 若模型和并行配置没问题，可以检查一下optimizer，设置yaml中的model.optim.name=distributed_fused_adam，将optimizer states切分到多张卡上。

Q: DistributedFusedAdam是ZeRO-2吗？
A: NeMO是调用的Apex中的[DistributedFusedAdam](https://github.com/NVIDIA/apex/blob/a7de60e57f0534266841e1733262601ad76aaa74/apex/contrib/optimizers/distributed_fused_adam.py#L272)，虽然Apex的实现是支持ZeRO-2，但NeMo中并没有暴露ZeRO-2的用法，现在只能用ZeRO-1，后续有计划支持ZeRO-2和FSDP。

Q: 如何profile？   
A: 若是用NeMo-Framework-Launcher的main.py启动，直接在yaml中设置`nsys_profile.enabled=True`。若是手动跑megatron_gpt_pretraining.py，需要在命令中加上nsys profile的命令，比如`CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nsys profile -s none -t nvtx,cuda -o /path/to/output --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 -u /path/to/megatron_gpt_pretraining.py  --config-path=xxx  --config-name=xxx`

Q: 如何确认是否成功开启TE?
A: 用nsys profile看是否有TE的kernel，比如FP8的gemm kernel。

Q: 无SLURM/K8S环境如何进行跑多卡测试？   
A: 每个节点手动跑torchrun，指定相同的master addr和master port。比如：
```
# master node, ip is 10.117.4.44
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=8888 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=localhost:8888 megatron_gpt_pretraining.py ...

# other nodes
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=10.117.4.44 --master_port=8888 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=10.117.4.44:8888 megatron_gpt_pretraining.py ...
```

## PaddleNLP
PaddleNLP+TE的测试方法参考https://github.com/Wong4j/PaddleNLP/blob/jaywan/te_integration/llm/llama/README_TE.md 

### 训练脚本和配置

### 收敛性测试
Q: FP8 recipe如何设置？
A: 目前PaddleNLP的分支中默认设置amax_history_len=1024，algo=max，是推荐的训练配置，收敛性测试时可不做修改。(NeMo的某些yaml中会默认设定amax_history_len=1，amax_algo=most_recent)。

Q: 训练开启TE和不开TE的曲线下降趋势不同
A: 目前PaddleNLP中对TE layer的参数初始化可能没有完全对齐非TE的layer，所以冷启动会出现TE和非TE的loss曲线不太吻合。解决办法是用[README_TE](https://github.com/Wong4j/PaddleNLP/blob/jaywan/te_integration/llm/llama/README_TE.md)中的checkpoint converter。先不开TE，训练一步，保存ckpt，然后转成TE的ckpt。将两份ckpt分别保存在两个目录中，作为初始的weight来热启动训练，这样就能完全对齐初始化weight，loss曲线可以几乎完全吻合。

Q: checkpoint converter可以分布式跑吗？
A: 不行，只是一个单线程的Python脚本。PaddleNLP的分布式训练ckpt默认会保存在同一个目录下，有多个`.pdparams`和`.opt`等文件，ckpt converter仅处理后缀为`.pdparams`的文件，将所有的参数转成TE对应的参数名，并对其中特定的参数做转置等变换，然后保存到另一个目录中。注意，不同的并行配置，其ckpt不能通用，比如pipeline parallel=4，那么目录下会有4个后缀为`.pdparams`的文件，分别是`model_state.pp00.pdparams, model_state.pp01.pdparams, ...`，如果下次训练不再使用pipeline parallel=4，就必须重新处理ckpt。



