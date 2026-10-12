# ST-CRF：基于步意图学习与条件随机场的跨域行人轨迹预测

[English](README.md)

论文 *ST-CRF: Cross-Domain Pedestrian Trajectory Prediction via Step-Intention Learning with Conditional
Random Fields*（Pengqian Han, Supratim Gupta, Jiamou Liu, Prakash Ranjitkar, Partha Roop；已投稿 IEEE Access）
的官方代码和训练好的模型。

ST-CRF 为行人轨迹的每一个观测步和未来步标注横向（直行 / 左转 / 右转）和纵向（匀速 / 减速 / 加速）的
**步意图**，标签由轨迹自动计算。一条线性链 CRF 作用在步意图序列上，监督共享的社交 Transformer 编码器；
预测步的后验边缘概率生成**由步意图驱动的锚点轨迹**，再由残差头在 SingularTrajectory 的低秩奇异空间中修正。
步意图标签只在训练时使用。

## 结果

ETH/UCY 上 20 个单场景跨场景迁移的平均 ADE/FDE（米），单条预测，按 SingularTrajectory 的协议和数据文件评测
（5 个随机种子的均值）：

| 模型 | 参数量 | 只用轨迹 | 使用场景地图 |
|---|---:|---:|---:|
| SingularTrajectory | 1,428,512 | 0.54/1.15（去掉地图后重新训练） | 0.48/0.94 |
| **ST-CRF** | **316,826** | **0.52/1.12** | **0.47/0.93** |

论文中还报告了：用单个 ETH/UCY 场景训练、在 SDD 上测试的结果（0.64/1.30）；标准 leave-one-out 协议
（只用轨迹 0.54/1.12，使用地图 0.48/0.95；SingularTrajectory 为 0.47/0.90）；以及 20 条预测的多模态
leave-one-out（best-of-20，0.21/0.37 和 0.21/0.34）。

与 SingularTrajectory 相同，跨域实验的 checkpoint 在目标场景的验证集上选择；梯度更新不使用目标场景的任何
文件。leave-one-out 模型完全不使用测试场景的数据。

## 安装

```bash
git clone https://github.com/pengqianhan/ST-CRF-code.git && cd ST-CRF-code
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # 测试环境：Python 3.10，PyTorch 2.5.1（CUDA 12.1）
```

## 数据

ST-CRF 使用 SingularTrajectory 的数据集和 Social-Implicit 的 SDD 文件。数据默认放在 `./datasets`，
也可以用环境变量 `STCRF_DATA` 指定其他目录。

1. **ETH/UCY、跨场景迁移对、单应矩阵、场景图像和地图**（来自
   [SingularTrajectory](https://github.com/InhwanBae/SingularTrajectory)）：
   ```bash
   git clone https://github.com/InhwanBae/SingularTrajectory.git /tmp/SingularTrajectory
   (cd /tmp/SingularTrajectory && ./script/download_datasets.sh)
   mv /tmp/SingularTrajectory/datasets ./datasets
   ```
2. **ETH/UCY 格式的 SDD**（来自 [Social-Implicit](https://github.com/abduallahmohamed/Social-Implicit)），
   只有表 5 需要：
   ```bash
   git clone https://github.com/abduallahmohamed/Social-Implicit.git /tmp/Social-Implicit
   python scripts/prepare_sdd.py --sdd /tmp/Social-Implicit/datasets/sdd
   ```

准备好后的目录结构：

```
datasets/
  A2B/ ... E2D/          20 个跨场景迁移对（A-E = ETH、Hotel、Univ、Zara1、Zara2），各含 train/ val/ test/
  A2S/ ... E2S/          SDD 迁移对（train = 源场景，val/test = SDD）
  eth/ hotel/ univ/ zara1/ zara2/   leave-one-out 划分
  homography/ image/ vectorfield/   使用场景地图的版本所需的地图
```

处理后的数据在第一次使用时缓存到 `./cache`。

## 预训练模型（v1.0 release）

[v1.0 release](https://github.com/pengqianhan/ST-CRF-code/releases/tag/v1.0) 包含论文中全部模型的权重，分为三个压缩包，
每个压缩包都解压到 `checkpoints/<实验名>/<迁移对>_s<种子>.pt`。

| 文件 | 包含的实验 | 用于 | 大小 |
|---|---|---|---:|
| `stcrf-checkpoints-main.zip` | `full_traj_lam03`、`full_mapf_lam03`、`none_traj`、`none_mapf` | 表 3、4、5、6 | 584 MB |
| `stcrf-checkpoints-ablation.zip` | `fullce_*`、`obscrf_*`、`crf2_*` | 表 4、RQ5 | 520 MB |
| `stcrf-checkpoints-multimodal.zip` | `stoch_*` | 表 7 | 74 MB |
| `SHA256SUMS.txt` | 三个压缩包的校验值 | | |

快速开始（在仓库根目录执行）：

```bash
URL=https://github.com/pengqianhan/ST-CRF-code/releases/download/v1.0
for f in stcrf-checkpoints-main.zip stcrf-checkpoints-ablation.zip stcrf-checkpoints-multimodal.zip SHA256SUMS.txt; do
  wget -nc "$URL/$f"
done
sha256sum -c SHA256SUMS.txt          # 每一行都应显示 OK
for f in stcrf-checkpoints-*.zip; do unzip -q -n "$f"; done
ls checkpoints                       # 应有 14 个实验文件夹
```

也可以只下载需要的压缩包：复现表 3、5、6 只需要主模型包。没有下载全部压缩包时，用
`sha256sum -c SHA256SUMS.txt --ignore-missing` 校验。

## 用权重复现论文表格

请先准备数据（见[数据](#数据)）。`evaluate.py` 在每个 checkpoint 对应迁移对的测试集上评测，写入
`results/<实验名>/<迁移对>_s<种子>.json`；`summarize.py` 输出各迁移对按种子取平均后的总体平均值，以及该平均值在种子之间的
标准差。整个过程不需要训练，有 GPU 时自动使用 GPU。给 `summarize.py` 加上 `--pairs`，会按论文表格的列顺序
（E2H、E2U、…、Z22Z1）打印每个迁移对的结果。

下面的预期输出是论文中训练时得到的测试结果；用权重重新评测，应得到相同的数值（最后一位可能有舍入差异）。

**表 3，以及表 4 的第一行和最后一行**（主模型包；20 个跨域迁移对，5 个种子）：

```bash
python evaluate.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf
python summarize.py full_traj_lam03 full_mapf_lam03 none_traj none_mapf
```
```
experiment           seeds  average ADE/FDE  std over seeds
full_traj_lam03          5    0.519/1.119     0.004/0.011
full_mapf_lam03          5    0.472/0.934     0.003/0.002
none_traj                5    0.528/1.132     0.002/0.006
none_mapf                5    0.480/0.966     0.002/0.003
```

**表 4 的消融行**（消融包；3 个种子）：

```bash
python evaluate.py --exps fullce_traj_l03 fullce_mapf_l03 obscrf_traj_l03 obscrf_mapf_l03 crf2_traj_lam03 crf2_mapf_l03
python summarize.py fullce_traj_l03 fullce_mapf_l03 obscrf_traj_l03 obscrf_mapf_l03 crf2_traj_lam03 crf2_mapf_l03
```
```
fullce_traj_l03          3    0.516/1.110     0.002/0.009
fullce_mapf_l03          3    0.476/0.941     0.000/0.003
obscrf_traj_l03          3    0.521/1.118     0.001/0.005
obscrf_mapf_l03          3    0.474/0.946     0.001/0.003
crf2_traj_lam03          3    0.519/1.118     0.002/0.007
crf2_mapf_l03            3    0.473/0.936     0.002/0.004
```

**表 5，SDD**（主模型包；需要先准备 SDD 迁移对，见[数据](#数据)；5 个种子）：

```bash
python evaluate.py --exps full_traj_lam03 none_traj --pairs A2S B2S C2S D2S E2S
python summarize.py full_traj_lam03 none_traj --sdd
```
```
full_traj_lam03          5    0.641/1.300     0.003/0.010
none_traj                5    0.639/1.301     0.000/0.004
```

**表 6，leave-one-out**（主模型包；3 个种子；ST-CRF 的结果在表 6 中，不使用步意图监督的版本在 RQ5 正文中引用）：

```bash
python evaluate.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf --pairs eth hotel univ zara1 zara2
python summarize.py full_traj_lam03 full_mapf_lam03 none_traj none_mapf --loo
```
```
full_traj_lam03          3    0.537/1.123     0.007/0.013
full_mapf_lam03          3    0.484/0.953     0.003/0.013
none_traj                3    0.535/1.114     0.002/0.001
none_mapf                3    0.479/0.930     0.003/0.001
```

leave-one-out 下的消融：用消融包中的实验名，加上 `--pairs eth hotel univ zara1 zara2` 和 `--loo` 即可。

**表 7，多模态 leave-one-out，best-of-20**（多模态包；3 个种子）：

```bash
python evaluate.py --exps stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --pairs eth hotel univ zara1 zara2
python summarize.py stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --loo
```
```
stoch_full_traj          3    0.214/0.367     0.002/0.006
stoch_full_mapf          3    0.205/0.344     0.002/0.003
stoch_none_traj          3    0.216/0.369     0.003/0.007
stoch_none_mapf          3    0.206/0.335     0.001/0.004
```

实验名与论文中名称的对应：

| 实验名 | 论文中的名称 |
|---|---|
| `full_traj_lam03`、`full_mapf_lam03` | ST-CRF（只用轨迹 / 使用场景地图） |
| `none_traj`、`none_mapf` | 不使用步意图监督（w/o step-intention） |
| `fullce_traj_l03`、`fullce_mapf_l03` | CE (hist. + pred.) |
| `obscrf_traj_l03`、`obscrf_mapf_l03` | CRF (hist. only) |
| `crf2_traj_lam03`、`crf2_mapf_l03` | CRF (pred. only) |
| `stoch_full_*`、`stoch_none_*` | 多模态 leave-one-out（S = 20），使用 / 不使用步意图监督 |

各实验的配置在 `experiments.py` 中；每个迁移对的设置（速度阈值、批大小、学习率调度）取自 SingularTrajectory
公开的配置（`configs/`）。

## 训练

```bash
# 单个迁移对、单个种子
python train.py --pair A2B --seed 0 --cfg '{"intent": "full_crf", "avg2": true, "lam": 0.3}'
# 多个迁移对和种子，每块 GPU 并行多个任务；已完成的任务会跳过
python run_all.py --exps full_traj_lam03 --seeds 0 1 2 3 4 --gpus 0 --per_gpu 4
# 训练论文中的全部模型
sh scripts/train_paper.sh
```

`run_all.py` 把 checkpoint 写入 `checkpoints/`、测试结果写入 `results/`，所以重新训练的模型同样可以用
`summarize.py` 汇总。

## 分析脚本

| 脚本 | 论文中的内容 |
|---|---|
| `scripts/crf_transitions.py` | 学到的转移分数，以及 Viterbi 解码与逐步取最大值的差异（RQ3） |
| `scripts/fig23_values.py` | 图 2 和图 3 中的数值 |
| `scripts/intent_eval.py` | 预测步意图的准确率（RQ3） |

## 目录结构

```
stcrf/        模型（model.py）、线性链 CRF（crf.py）、步意图标签（intention.py）、
              归一化与奇异空间（geometry.py）、场景地图修正（mapadjust.py）、数据（data.py）
configs/      取自 SingularTrajectory 公开配置的各迁移对设置；SDD 的设置
train.py      单个迁移对的训练和评测
run_all.py    在多块 GPU 上运行多个迁移对和种子
evaluate.py   评测保存的 checkpoint
summarize.py  按迁移对和总体对种子取平均
```

## 许可证与致谢

轨迹归一化、奇异空间、数据加载和场景地图修正沿用
[SingularTrajectory](https://github.com/InhwanBae/SingularTrajectory)（Bae et al., CVPR 2024）的做法，
`stcrf/data.py`、`stcrf/geometry.py` 和 `stcrf/mapadjust.py` 的部分代码改写自其代码。SingularTrajectory 采用
Creative Commons Attribution-NonCommercial 4.0 International（署名—非商业性使用 4.0 国际）许可证，本仓库采用
相同的许可证（[LICENSE](LICENSE)）。ETH/UCY 格式的 SDD 文件来自
[Social-Implicit](https://github.com/abduallahmohamed/Social-Implicit)。

## 引用

```bibtex
@article{han2026stcrf,
  title   = {{ST-CRF}: Cross-Domain Pedestrian Trajectory Prediction via Step-Intention Learning with Conditional Random Fields},
  author  = {Han, Pengqian and Gupta, Supratim and Liu, Jiamou and Ranjitkar, Prakash and Roop, Partha},
  journal = {IEEE Access},
  note    = {under review},
  year    = {2026}
}
```
