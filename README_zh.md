# FocusEdit

**无需训练的空间调制引导，用于保持背景的真实图像编辑。**

[English](README.md) · [实验复现](docs/REPRODUCIBILITY.md) · [方法说明](docs/METHOD.md)

FocusEdit 从 SD3 内部的编辑词注意力中预测编辑区域，再对目标条件与源条件的速度差进行空间调制。默认使用二值区域图和 0.2 的背景引导下限，无需训练额外的定位模型。

本仓库以 `FocusEditPipeline` 为统一入口，包含定位、引导、完整实验配置、统一条件下的基线、逐例指标和统计复算脚本。底层编辑过程基于 Navi-FlowAlign，相关来源及许可保留在独立的说明中。

## 使用

Python 3.11，安装适合 GPU 的 PyTorch / torchvision 后运行：

```bash
python -m pip install -e .
focusedit --image input.png \
  --source-prompt "A red car" \
  --target-prompt "A blue car" \
  --output outputs/edited.png
```

默认配置为 SD3 Medium、512×512、BF16、50 个调度步、50 个编辑步、参考深度 42、CFG 13.5。权重需从模型提供方取得并接受相应许可。`--model` 支持自备权重目录。

## 核对结果

```bash
python results/recompute_means.py
```

无需 GPU，可以从 9,800 条指标记录复算全部 84 项均值与有效样本数。补充实验覆盖全部 700 个编辑样本，背景指标使用其中 556 个非空背景样本。默认 FocusEdit 的背景 LPIPS 相对统一设置下的未门控基线降低 57.4%（VAE 参考）和 53.8%（RGB 参考）；CLIP-edit 为 22.823，基线为 23.086。所有配置完整保存在结果文件中。

本次补充实验与原投稿表格分别标注，完整的配置、评测口径和适用范围见英文 README 与实验文档。
