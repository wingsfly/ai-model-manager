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
aim link            登记外部应用/软链对模型的依赖(可 --scan 自动发现)
aim unlink          移除外部依赖登记
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

### aim ingest — 原生模型摇入 store

把 HuggingFace / Ollama / ModelScope 的原生缓存模型「摇入」`store/{类别}/{id}/` 扁平单目录，并在工具原缓存位置重建「加载壳」指回 store，使工具仍能原生加载；同时在 registry 写入 `storage` 标注（供备份/还原与 `aim verify --fix` 重建壳）。**取代旧的 `aim convert`**（旧实现会把缓存复制成 CAS 结构、体积翻倍且工具无法加载）。

```bash
aim ingest <model_id>            # 摇入单个原生模型
aim ingest --all-native          # 摇入所有 native_cas 模型(HF/Ollama/MS)
aim ingest <model_id> --dry-run  # 预览，不改任何文件
aim ingest <model_id> --keep-native   # 保留原生字节(默认回收)
aim ingest <model_id> --new-id NEW --category CAT
```

**三种工具的壳：**
- **HuggingFace**：`{HF_HOME}/hub/models--org--repo/` 的 `snapshots/<commit>/<file>` 重建为指向 store 的绝对软链（删除 `blobs/`，除非 `--keep-native`）；`from_pretrained("org/repo")` 照常。
- **Ollama**：GGUF 以硬链接共享进 store（同 inode，缓存不动）；小 blob 与 manifest 复制进 store 作元数据。`ollama run` 照常。
- **ModelScope**：缓存里的模型目录换成指向 store 的目录软链（含 `.msc/.mdl/.mv` 元数据）。

**安全：** copy 优先，壳建好前原生字节始终完好；失败自动回滚，绝不丢数据；`--dry-run` 不落盘。

### aim convert（已弃用）

`aim convert` 现在是 `aim ingest` 的弃用别名，会打印提示并委托给 ingest。请改用 `aim ingest`。

### aim verify --fix（扩展）

除原有链接校验外，`aim verify` 现在还校验每个已摇入模型的「加载壳」是否仍指向 store；`aim verify --fix` 会**用 `storage` 标注重建丢失/损坏的壳**（HF snapshots、Ollama blobs+manifest、MS 目录软链），是「壳可再生」的自愈能力，也预演了备份还原（SP3）。

---

### aim backup / restore — 可移植备份与还原

把整库备份到目录(外置盘/另一路径),并能在换机/换盘上一键还原。

```bash
aim backup <dir>            # 把 store/ 镜像到 <dir>/store/ + 写 <dir>/aim-backup.json(幂等、可重跑)
aim backup <dir> --verify   # 用 quick-hash 比对(而非仅大小)
aim restore <dir>           # 重建 store、导入 registry、为本机重算并重建 HF/Ollama/MS 加载壳;检测并打印建议 env
aim restore <dir> --apply-env  # 额外把 env 写入 shell 配置
aim restore <dir> --root <id>  # 还原到指定存储根(默认 primary)
```

**说明:**
- 备份只含 store 真实字节 + 一个小 JSON 清单;**加载壳与各工具缓存不入备份**(还原时从 `storage` 标注重建),适合几百 GB 库换机/换盘。
- 还原**按目标机的缓存路径重算壳位置**(读目标机 HF_HOME/OLLAMA_MODELS/MODELSCOPE_CACHE),所以源机与目标机路径不同也能正确落位。
- 还原默认**不改 shell 配置**(只打印建议 env);`--apply-env` 才写。幂等可重跑;单个模型壳重建失败不中断整库还原、末尾汇总。
- 若存在尚未 `aim ingest` 的原生模型,备份会告警(它们不在 store、不被备份)。

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

### aim link / unlink — 登记外部依赖

`provision` 记录的是 **aim 自己**为 9 个引擎建的链接（都在根内）。但根**外**的应用常通过手动软链消费 store 里的模型（如 `~/.cache/dolphin → store`、`~/.cache/huggingface → ~/AI/huggingface`）——aim 原本对此一无所知，`migrate`/`delete` 会悄悄弄断它们。`aim link` 把这些外部依赖登记进模型的 `external_links`，于是 `info` 能看到、`verify` 会校验、`delete` 会警告、`migrate` 会**自动重指**外部软链到新位置。

```bash
# 登记一个已存在的外部软链/路径对模型的依赖
aim link ms-dataoceanai-dolphin-small ~/.cache/dolphin/small --consumer "OpenSpeechAPI dolphin"

# 登记并顺便创建软链(external_path → 模型 store 路径)
aim link <model_id> ~/.cache/foo/bar --consumer myapp --create

# 只记录依赖、不建实际链接(程序按路径/配置加载)
aim link <model_id> /opt/app/models/x --type reference --consumer myapp

# 自动发现:扫描缓存目录里指向 aim 根的软链并批量登记
aim link --scan                 # 预览(dry-run)
aim link --scan --apply         # 实际登记
aim link --scan --scan-roots ~/.cache,~/Library/Caches --apply

# 移除登记(加 --remove 连软链一起删)
aim unlink <model_id> ~/.cache/foo/bar
aim unlink <model_id> ~/.cache/foo/bar --remove
```

**参数：** `--consumer` 消费方名称；`--type symlink|hardlink|reference`；`--create` 顺便建链；`--scan [--apply] [--scan-roots ...]` 自动发现登记。

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

### aim env — 检测/管理下载源环境变量

aim 能检测各下载源(HuggingFace/Ollama/ModelScope/PyTorch Hub/Civitai/Git)的环境变量配置——读 shell 启动文件、登录 shell 回显、或工具命令——并以「aim 独占 env 文件 + rc 仅加一行受 marker 保护的 source」的方式管理。

```bash
aim env show              # 检测并展示各源变量(当前值/来源/状态)+ 解析出的缓存目录(只读)
aim env show --json       # 机器可读;secret 值已脱敏
aim env path huggingface  # 打印某源解析后的缓存目录(脚本用)
aim env apply --shell zsh # 写 ~/.aim/env.{sh,fish},并在 rc 插入一行受 marker 保护的 source
aim env apply --set HF_ENDPOINT=https://hf-mirror.com --set HF_HUB_ENABLE_HF_TRANSFER=1
aim env apply --dry-run   # 预览,不写任何文件
aim env apply --service   # 额外打印 ollama 等服务级 env 设置命令(launchctl/systemd)
```

**说明:**
- 检测会把各源真实缓存位置写回 `~/.aim/config.json` 的 `sources.<源>.cache_path`,使 `aim scan` 即使在 HF/MS 缓存位于默认 `~/.cache` 时也能扫到模型。
- secret(如 `HF_TOKEN`/`CIVITAI_API_TOKEN`)不会写入 env 文件或 rc;优先用工具原生机制,兜底写入 `~/.aim/secrets.env`(权限 600)。
- 缓存目录的「搬迁」不在此命令范围(属后续阶段);本命令默认采纳现状,只主动管理 endpoint/加速等安全变量。
- 跨 shell:zsh/bash/fish 分别生成对应格式并接线到对应启动文件;改动前自动备份 `<rc>.aim.bak`。

### aim sources — 下载源与工具

```bash
aim sources list                      # 列出所有源、其下载工具安装状态、env 概览
aim sources list --json
aim sources install huggingface -y    # 安装该源的下载工具(复用后台工具自动安装)
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
