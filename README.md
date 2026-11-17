# CS5185-G30 | Content-Adaptive Video Encoding

## 三人任务分工

| 成员 | 工作 | 交付物 |
| --- | --- | --- |
| A（本分支） | 编码与集成。读取元数据，计算 5% / 2% 文件预算，运行统一基线和自适应入口，执行 FFmpeg x264 两遍编码、超预算重试和播放完整性验证。最终与 B、C 联调。 | 编码 CLI、运行说明、编码与重试日志、报告中的系统流程与反馈调整、对应汇报内容。 |
| B | 内容特征与选参。计算帧差和边缘密度，绘制时间曲线，在开发集比较参数并冻结规则。 | 选参模块、配置、曲线、一个片段的选参解释、报告中的特征和策略。 |
| C | 画质与实验。用原片对齐评估 PSNR 和 MS-SSIM，分析两档的基线与自适应结果，组织主观看片。 | 评估模块、实验表、对比截图或短片，汇总三人的报告 PDF 和约 5–6 页幻灯片。 |

开发集为 01、02、03、04、07、08，最后测试保留 05、06、09、10。11、12 在策略冻结后作为补充案例运行。最终运行 12 段视频各两档，详细展示至少 3 段。三人一起核对产出，一位组员提交含代码、报告 PDF、实验结果、幻灯片的 ZIP。

## 运行编码

需要 Python 3.10+，以及带有 `libx264` 的 FFmpeg（支持 `-fps_mode`）和同套发行版的 ffprobe。A 模块只用 Python 标准库。在终端运行 `ffmpeg -version`、`ffprobe -version` 和 `ffmpeg -hide_banner -h encoder=libx264` 确认可用。也可以通过 `--ffmpeg`、`--ffprobe` 指定路径。

从课程 ZIP 提取数据集。输入是单个 MP4 或直接包含 MP4 的目录，使用课程无音轨、1280×720、30 fps 输入。目录名和路径可以包含空格。每次运行指定一个新输出目录。

```powershell
python encode.py "path/to/CS5185_Project_Video_Dataset" --output "runs/baseline"
python encode.py "path/to/01_FourPeople.mp4" --output "runs/smoke"
```

基线使用 `medium`、AQ mode 1，两遍编码的输入均为原片。预算按原文件实际字节数向下取整。初始视频码率预留 2% 容器余量，超限时根据目标大小与实际大小比例下调码率。默认最多 3 次，每次包含两遍编码。最终 MP4 必须通过 FFmpeg 完整解码检查，且分辨率、帧率、帧数、时长和相对时间戳通过核对。满足条件后才发布最终 MP4。

输出位于 `<run>/<method>/<5|2>/<video>.mp4`。`run.json` 记录工具版本和运行设置，`jobs.jsonl` 每行记录一个任务，包含预算、尝试参数和命令、大小、时间、调整原因、输出路径及失败原因。实际视频码率按视频包字节数计算，文件码率包括容器开销。存在失败任务时程序仍处理其余任务，最终退出码为 1。

## B、C 的集成接口

把两个模块放在仓库根目录，或放在 Python 可导入的路径下。模块名由命令行指定。

- **B** 提供 `select(source: pathlib.Path, metadata: dict, target_bytes: int) -> dict`。元数据包含 `bytes`、`duration_s`、`width`、`height`、`fps`（分数字符串）、`frames` 和 `codec`。返回字典至少包含 `preset`（x264 preset 字符串）和 `aq_mode`（0–3 的整数）。可一并返回 `reason`、`features` 等可 JSON 序列化字段，程序会把完整决策写入任务日志。
- **C** 提供 `evaluate(source: pathlib.Path, encoded: pathlib.Path) -> dict`。在合规且完整的候选 MP4 上计算画质，并返回可 JSON 序列化的字段，例如 `psnr_y_db`、`ms_ssim`。如果 PSNR 为无穷大，使用字符串 `"inf"`。编码候选路径仅在回调期间可用。

接入两个模块后，用同一流程比较基线与自适应结果。

```powershell
python encode.py "path/to/CS5185_Project_Video_Dataset" --output "runs/comparison" --mode both --selector features:select --evaluator quality:evaluate
```

报告与展示中应注明课程数据集和素材来源。数据集的 `licenses/` 目录提供各视频的原始署名资料。
