# aim — AI Model Manager 使用手册

统一管理 `~/AI/` 下所有 AI 模型的 CLI 工具。支持 Ollama、HuggingFace、oMLX、ComfyUI、Whisper、Coqui-TTS、SparkTTS、Piper、Fish-Speech 共 9 个引擎。

## 安装

```bash
# 已自动安装，确认可用：
aim --version   # aim 0.1.0
which aim       # ~/.local/bin/aim → ~/AI/aim.py
```

## 核心概念

| 概念 | 说明 |
|------|------|
| **Canonical Store** | `~/AI/store/` — 模型的唯一真实副本存放处 |
| **Provision** | 引擎目录中指向 store 的链接（hardlink / symlink） |
| **Native CAS** | Ollama / HuggingFace 自带内容寻址系统，不迁入 store |
| **Root** | 存储根目录，支持多磁盘（主 SSD、外置硬盘等） |
| **Registry** | `~/AI/.aim/registry.json` — 所有模型的元数据索引 |

### 链接策略

| 场景 | 链接类型 | 原因 |
|------|---------|------|
| 同卷 + 文件 | hardlink | APFS 原生支持，无悬挂风险 |
| 同卷 + 目录 | symlink | macOS 不支持目录 hardlink |
| 跨卷 | symlink | hardlink 不能跨文件系统 |

### 模型分类

```
llm/       chat, code, embedding, vision
image-gen/ checkpoint, lora, vae, controlnet, text-encoder, unet, upscaler
tts/       model, vocoder, voice
asr/       model
audio/     codec, vad
```

---

## 命令速查

```
aim scan            扫描引擎目录，发现并注册模型
aim list            列出已注册模型
aim info            查看模型详情
aim status          存储概览

aim download        下载模型
aim provision       为引擎创建链接
aim unprovision     移除引擎链接
aim update          检查/执行更新
aim delete          删除模型

aim root add/list   管理存储根目录
aim migrate         迁移模型到其他根

aim dedup           扫描/修复重复文件
aim verify          验证链接完整性
aim orphans         发现未注册文件
aim config show     查看配置
```

---

## 命令详解

### aim scan — 扫描引擎目录

扫描所有（或指定）引擎目录，将发现的模型注册到 `registry.json`。

```bash
aim scan                  # 扫描全部 9 个引擎
aim scan --engine omlx    # 仅扫描 oMLX
aim scan --engine comfyui # 仅扫描 ComfyUI
```

首次使用 aim 时必须先运行一次全量扫描。后续新增模型后重新扫描即可增量更新。

---

### aim list — 列出模型

```bash
aim list                              # 列出所有模型
aim list --engine comfyui             # 仅 ComfyUI 中的模型
aim list --category tts               # TTS 相关模型（前缀匹配）
aim list --category image-gen/lora    # 精确到子类别
aim list --format safetensors         # 按格式筛选
aim list --sort size                  # 按大小降序
aim list --sort name                  # 按名称排序（默认）
aim list --provisions                 # 显示每个模型的链接详情
aim list --for comfyui                # 可用于 ComfyUI 的模型
```

**组合使用：**
```bash
aim list --category image-gen --sort size --provisions
aim list --engine omlx --format mlx-safetensors
```

---

### aim info — 查看模型详情

```bash
aim info flux1-dev                # 查看 FLUX.1-dev 的完整信息
aim info qwen3.5-27b-8bit        # 查看 Qwen3.5-27B
aim info spark-tts-0.5b --provisions  # 包含链接详情
```

输出包括：ID、名称、分类、格式、大小、来源、canonical 路径、是否 native CAS、标签、添加时间、所有 provisions。

如果 model_id 不完全匹配，会尝试模糊搜索并给出建议。

---

### aim status — 存储概览

```bash
aim status                # 默认按引擎分组
aim status --by engine    # 按引擎分组（默认）
aim status --by category  # 按模型类别分组
aim status --by root      # 按存储根分组
```

显示：磁盘总量/已用/可用、模型总数和总大小、各分组的模型数和占用空间。

---

### aim download — 下载模型

```bash
# 从 HuggingFace 下载
aim download hf:black-forest-labs/FLUX.1-dev
aim download hf:Qwen/Qwen3-TTS-12Hz-1.7B-Base --name qwen3-tts --category tts/model

# 从 Ollama 拉取
aim download ollama:gemma3n:latest
aim download ollama:llama3:8b

# 从 URL 直接下载
aim download url:https://example.com/model.safetensors --name my-model

# 从 ModelScope 下载
aim download ms:Qwen/Qwen2-7B --category llm/chat
```

**参数：**
| 参数 | 说明 |
|------|------|
| `source` | 来源，格式见上方示例 |
| `--name ID` | 自定义模型 ID（默认从 source 推断） |
| `--category CAT` | 模型分类（如 `llm/chat`, `tts/model`） |

下载的 HuggingFace 模型默认使用 `~/AI/hfd.sh`（hfd 工具），不可用时回退到 `huggingface-cli`。

---

### aim provision — 为引擎创建链接

将 store 中的模型链接到引擎目录，使引擎可以直接使用。

```bash
# 为 oMLX 创建 symlink
aim provision qwen3.5-27b-8bit --engine omlx

# 为 ComfyUI 创建 hardlink，指定子目录
aim provision flux1-dev --engine comfyui --subdir checkpoints
aim provision flux1-dev --engine comfyui --subdir unet

# 为 Whisper 创建链接
aim provision whisper-large-v3 --engine whisper
```

**注意：** Native CAS 引擎（Ollama、HuggingFace）不需要 provision。

---

### aim unprovision — 移除引擎链接

```bash
aim unprovision flux1-dev --engine comfyui    # 移除 ComfyUI 中的链接
aim unprovision qwen3.5-27b-8bit --engine omlx
```

仅移除链接，不删除 store 中的模型文件。

---

### aim update — 检查更新

```bash
aim update                    # 列出所有可更新的模型
aim update flux1-dev          # 查看特定模型的更新信息
aim update --check            # 仅检查，不执行
aim update --all              # 检查所有模型
```

目前显示各模型的来源信息，实际更新请使用对应引擎工具（`ollama pull`、`hfd.sh` 等）。

---

### aim delete — 删除模型

```bash
aim delete whisper-large-v3           # 删除模型及所有链接
aim delete gemma3n --force            # 强制删除 native CAS 模型
```

**流程：** 移除所有 provisions（链接）→ 删除 store 中的副本 → 从 registry 中移除。

Native CAS 模型（Ollama/HuggingFace）默认拒绝删除，需要 `--force`。

---

### aim root — 管理存储根

```bash
aim root list                                    # 列出所有根及磁盘空间
aim root add /Volumes/ExternalSSD/AI --label "External SSD"  # 添加新根
```

添加新根会自动创建 `.aim/` 和 `store/` 子目录。

---

### aim migrate — 迁移模型

将模型从一个存储根迁移到另一个。

```bash
# 迁移单个模型
aim migrate whisper-large-v3 --to ext-ssd

# 按类别批量迁移
aim migrate --category image-gen --to ext-ssd
aim migrate --category tts --to ext-ssd
```

**流程：** 复制到目标 root 的 store → 更新 registry → 重建链接 → 删除旧副本。

Native CAS 模型不支持迁移。

---

### aim dedup — 去重

```bash
aim dedup --scan    # 扫描重复文件（只读）
aim dedup --apply   # 执行去重（用 hardlink 替换重复副本）
```

扫描 >100MB 的文件，通过"文件大小 + 前 64KB 哈希"快速比对。去重时保留第一份副本，将同卷的其余副本替换为 hardlink。

---

### aim verify — 验证链接完整性

```bash
aim verify          # 检查所有 provision 链接是否有效
aim verify --fix    # 自动修复断裂的链接
```

检测：缺失的链接、悬挂的 symlink、指向错误目标的链接、未硬链接的文件。

---

### aim orphans — 发现未注册文件

```bash
aim orphans                   # 全部引擎
aim orphans --engine comfyui  # 仅检查 ComfyUI
```

查找引擎目录中存在但未被 registry 跟踪的模型文件。自动排除 `.venv`、`.git`、`node_modules`、`__pycache__` 等目录。

---

### aim config show — 查看配置

```bash
aim config show    # 输出当前 config.json
```

---

## 全局选项

```bash
aim --version          # 显示版本号
aim --root /path/to/AI # 覆盖默认根目录（默认 ~/AI）
```

---

## 典型工作流

### 初始化

```bash
aim scan          # 首次全量扫描
aim status        # 查看存储概览
aim list          # 查看所有模型
```

### 下载新模型并分发

```bash
aim download hf:org/model --name my-model --category llm/chat
aim provision my-model --engine omlx
aim provision my-model --engine comfyui --subdir checkpoints
```

### 去重节省空间

```bash
aim dedup --scan      # 查看重复
aim dedup --apply     # 执行去重
```

### 多磁盘管理

```bash
aim root add /Volumes/ExternalSSD/AI --label "External SSD"
aim root list
aim migrate --category image-gen --to external-ssd
aim verify
```

### 日常维护

```bash
aim scan              # 重新扫描发现新模型
aim verify --fix      # 修复断裂链接
aim orphans           # 清理未注册文件
```

---

## 配置文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 工具配置 | `~/AI/.aim/config.json` | 存储根、引擎配置、默认值 |
| 模型注册表 | `~/AI/.aim/registry.json` | 所有模型的元数据（自动备份为 .bak） |
| 模型存储 | `~/AI/store/` | Canonical Store 目录 |

### 支持的引擎

| 引擎 | 目录 | Native CAS | 原生格式 |
|------|------|:----------:|---------|
| ollama | `ollama/models/` | ✓ | gguf |
| huggingface | `huggingface/hub/` | ✓ | safetensors, bin, pt |
| omlx | `omlx/` | | mlx-safetensors |
| comfyui | `ComfyUI/models/` | | safetensors, pt, ckpt |
| whisper | `whisper/` | | pt, ct2 |
| coqui | `Coqui-TTS/` | | pth |
| sparktts | `sparktts/pretrained_models/` | | safetensors |
| piper | `piper/` | | onnx |
| fish-speech | `services/fish-speech/` | | safetensors |

---

## 依赖

**零外部依赖** — 仅使用 Python 3.10+ 标准库。
