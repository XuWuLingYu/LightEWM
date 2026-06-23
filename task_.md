当前主要任务, 其中3是重中之重: 
1. 用预训练权重跑通双向扩散模式的推理（验证环境 + 模型加载 OK）
2. 在 LIBERO 上跑一轮短微调（验证训练管线 OK）
3. 搭视频质量评测管线，用 FVD+SSIM+LPIPS 对三个权重做对比评测

## 环境与任务分析
2. 评测管线需要的环境准备：
视频质量评测（较简单）：
- GPU 服务器（推理用，单卡 A100 40GB+ 或 RTX 4090）
- pytorch-fvd / clean-fid 等指标计算库
- LIBERO 测试集的真实视频作为 ground truth
- 自己写评估脚本：批量推理 → 计算 FVD/SSIM/LPIPS

任务成功率评测（较复杂，当前做不了）：
- Linux 系统（MuJoCo 仿真器仅支持 Linux）
- robomimic==0.2.0 + robosuite==1.4.0 + gym==0.25.2（老版本三件套，安装有坑）
- MuJoCo 2.x（mujoco-py，编译问题多）
- LIBERO 仿真环境配置
- WAM actor 动作预测头（目前不存在，需要自己实现）
- 闭环控制接口：模型输出动作 → 仿真器执行 → 获取新观测 → 循环
3. 从"无敌好用"的视角看，除了评测管线还缺什么：
| 缺失模块 | 重要性 | 说明 |
|---------|--------|------|
| WAM actor（动作预测头） | ★★★★★ | 核心缺失。当前只能生成视频，不能输出机器人动作。加上后才是完整的世界动作模型 |
| 评测管线 | ★★★★★ | 如上所述，视频质量 + 任务成功率两套 |
| CALVIN 数据集支持 | ★★★☆☆ | Roadmap 中标记 TODO，CALVIN 是另一个主流基准，支持后适用范围更广 |
| 强化学习模块 | ★★★☆☆ | Roadmap 中标记 TODO，用于进一步优化策略质量 |
| 实时闭环控制接口 | ★★★★☆ | 当前推理输出是视频文件，没有实时动作流接口，无法用于在线控制 |
| 多数据集统一接口 | ★★★☆☆ | 目前只有 LIBERO 的转换脚本，其他数据集（RT-1/Bridge/DROID 等）需要自己写适配 |
| Causal-Forcing 文档 | ★★☆☆☆ | 因果模式依赖外部仓库 thu-ml/Causal-Forcing，但 README 完全没提 |
| 预训练权重评测数据 | ★★★★☆ | Model Zoo 有 4 个权重但无任何量化指标，用户无法判断效果好坏 |
| 一键训练/评测脚本 | ★★★☆☆ | 现有脚本需要手动改配置，缺少开箱即用的 end-to-end 脚本 |
总结：最关键的三块是 WAM actor + 评测管线 + 闭环控制接口，补齐这三块才能从"视频生成工具"变成"完整的机器人策略训练平台"。

## 部署
1. 批量推理不需要额外部署框架，但需要自己写批量调度脚本。
LightEWM 自带推理入口（scripts/infer.sh + WanInferRunner），是单次推理模式：给一张图+一段指令，生成一段视频。底层走的是 diffusers 的标准扩散采样流程，不需要 sGLang/vLLM 这类 LLM serving 框架（那些是给语言模型用的，视频扩散模型的推理模式不同）。
但现有脚本是"一次跑一个样本"的模式，要做批量评测需要自己包一层：
- 读取 LIBERO 测试集的所有（图片, 指令）对
- 循环调用推理，或改成 batch inference（需要改 WanInferRunner 支持 batch）
- 收集所有生成视频，统一计算指标
不需要 sGLang recipe，但需要写一个评测调度脚本（Python，几百行的量级）。

## 评测细节
Model Zoo（全部公开可下载）：
1. Wan2.2-TI2V-5B（基础模型）— ~50.5 GB
   HF: Wan-AI/Wan2.2-TI2V-5B
2. Wan2.2-5B-Robot（Robot 预训练权重）— ~9.3 GB
   HF: XuWuLingYu/Wan2.2-5B-Robot
   用途：机器人数据预训练的初始化权重，推荐作为微调起点
3. Wan2.2-5B-Libero（LIBERO 微调权重）— ~9.3 GB
   HF: XuWuLingYu/Wan2.2-5B-Libero
   用途：可直接推理，不需要训练
4. LIBERO-Causal-Wan2.2-5BTI2V（因果模型权重）— ~18.6 GB
   HF: XuWuLingYu/LIBERO-Causal-Wan2.2-5BTI2V

下载 Wan2.2-5B-Libero 权重 → 跑推理（不需训练）→ 看生成的视频质量是否合理

按权重类型分层评测：
| 权重 | 评测方法 | 指标 |
|------|---------|------|
| Wan2.2-5B-Libero（LIBERO微调） | 在 LIBERO 测试任务上推理生成视频，与真实演示视频对比 | FVD↓、SSIM↑、LPIPS↓、PSNR↑ |
| Wan2.2-5B-Robot（Robot预训练） | 同上，但预期效果弱于 Libero 微调版（未针对 LIBERO fine-tune） | 同上，作为 baseline |
| Wan2.2-TI2V-5B（原始基础模型） | 同上，作为最弱 baseline | 同上 |
推荐工具：
- FVD：pytorch-fvd 或 clean-fid（需要 I3D 特征提取）
- SSIM/PSNR：torchmetrics 或 scikit-image
- LPIPS：pip install lpips（仓库 pyproject.toml 里其实已经用到了 lpips，只是没声明依赖）
评测流程：对每个权重，在 LIBERO 的 4 个子集（Spatial/Object/Goal/Long）各采样 N 个任务推理，计算上述指标，形成对比表。这样能清晰看到预训练 → 微调的效果提升。

视频生成质量评测（当前可做）：
- 用 LIBERO 的测试任务做推理，生成视频
- 评估指标：FVD（视频分布距离）、SSIM、PSNR、LPIPS 等
- 对比生成视频与真实演示视频的视觉相似度
- 这个 LightEWM 当前就能做，只是缺评估代码需要自己补

任务执行成功率评测（当前做不了）：
- 在 LIBERO 仿真环境中实际执行动作，看机器人能否完成任务
- 评估指标：任务成功率（success rate）
- 这个需要 WAM actor（动作预测头），而 LightEWM 目前没有实现这个模块
- 所以当前只能评"看得准不准"，不能评"做得对不对"
