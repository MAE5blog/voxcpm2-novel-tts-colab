# VoxCPM2 长文本小说配音（免费 Colab）

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAE5blog/voxcpm2-novel-tts-colab/blob/main/VoxCPM2_Novel_TTS_Colab.ipynb)

面向单旁白、长篇小说的 Google Colab 工作流：代码从本仓库克隆，小说、参考音频、任务清单和成品留在 Google Drive。免费 Colab 断连或运行时被回收后，重新加载代码和模型即可从 `manifest.json` 继续。

## 使用方式

1. 点击上方 **Open in Colab**，在“运行时 → 更改运行时类型”选择 GPU（免费账户常见为 T4，但不保证可分配）。
2. 在 Google Drive 新建或使用 `MyDrive/VoxCPM2_Novel/inputs/`，把已获授权的参考音频放进去。首次测试默认文件名是 `古龙评书（干声）.flac`；它不会进入 GitHub。
3. 依次运行 Notebook。代码单元会克隆或同步本仓库，绝不会要求上传 `voxcpm_novel.py`。
4. 只修改 Notebook 的“输入与任务配置”单元：可使用仓库附带的原创长文本，或改成 Drive 中自己的 TXT / Markdown / EPUB / 有文字层 PDF。
5. 先运行“一段试听”，确认音色、速度和断句后，再运行全书生成与 M4B 导出。

以后更新项目代码，只需提交到本仓库；在 Colab 重新运行“获取/同步项目代码”单元即可。输入文件、音频、模型缓存和生成结果均不应提交。

## 仓库内容

- `VoxCPM2_Novel_TTS_Colab.ipynb`：中文 Colab 教程和唯一配置入口。
- `voxcpm_novel.py`：导入、中文切分、任务清单、断点续跑、OOM 二分、合并和 M4B 导出。
- `examples/generated_long_demo.txt`：原创长文本测试稿（9 章、约 8,583 个汉字）。
- `tests/test_voxcpm_novel.py`：不需要 GPU、VoxCPM 或 ffmpeg 的核心逻辑测试。
- `requirements-colab.txt`：Colab 的 Python 依赖；不升级 Colab 自带 PyTorch/CUDA。

持久化目录：

```text
MyDrive/VoxCPM2_Novel/
├── inputs/                         # 用户的小说和参考音频，不进 Git
└── jobs/<JOB_NAME>/
    ├── manifest.json                # 可续跑状态
    ├── segments/                    # 每段临时 WAV
    ├── chapters/                    # chapter_XXX.wav / .mp3
    └── exports/audiobook.m4b
```

## 设计边界

- 固定 `voxcpm==2.0.3`，默认 `load_denoiser=False`、`optimize=False`，适合免费 T4 的串行短段推理。
- 默认每段约 90 个中文字符、硬上限 160；`max_len=4096` 是音频 token 上限，不是文字数。
- T4 不支持模型默认 BF16 时，Notebook 只修改 `/content` 下的临时模型副本为 FP16；先试听是必须的运行时验收。
- 参考音频编码会尽量缓存；`normalize=True` 时会安全回退到普通生成路径。
- CUDA OOM 会按安全标点自动二分当前段，最多三层。真正卡死时重启 Colab，并用相同任务目录恢复。
- `manifest.json` 的签名包含小说、参数、参考音频哈希、转录和风格。改变其中任意项时，请改用新的 `JOB_NAME`。

这不是多角色 WebUI 或后台服务。若目标变为角色标注、可视化逐段编辑或长期服务化，可评估 [TTS-Story](https://github.com/Xerophayze/TTS-Story)、[abogen](https://github.com/denizsafak/abogen) 或 [Alexandria](https://github.com/Finrandojin/alexandria-audiobook)。

## 本地检查

```powershell
python -m py_compile .\voxcpm_novel.py
python -m unittest discover -s .\tests -v
```

## 许可与权利

项目采用 [AGPL-3.0-only](LICENSE)。`voxcpm_novel.py` 的派生来源说明见 [LICENSES.md](LICENSES.md)。VoxCPM2 模型、`voxcpm` 包及推荐项目各有其上游许可证和使用条款。

只使用你拥有或已获明确授权的文字、参考音频和声线；本仓库不包含、也不接受提交任何真实参考音频、书稿或生成的有声成品。
