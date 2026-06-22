# SP1 — 下载源/工具模型 + 环境变量管理（设计 spec）

- 日期：2026-06-22
- 状态：已批准（待写实现计划）
- 分支：`feat/aim-sp1-source-env-management`
- 作者：wingsfly + Claude

---

## 0. 整体架构上下文（为何如此）

本 spec 是一个更大目标的第一层。整体目标：让 aim 把 HF / Ollama / ModelScope / URL / PyTorch Hub / Civitai / Git
等下载源统一管理起来，做到「模型按 category + 单模型目录纳入 store」「各工具仍能原生加载」「整库可移植备份还原」。

**核心架构原则（已与用户确认）**：`~/AI/store/{category}/{model_id}/` 是模型的唯一真实副本（扁平单模型目录）；
每个工具通过一个**重建出来的、工具原生的「加载壳」**指回 store 来加载；每个模型带一份 manifest 标注，
记录存储类别 + 加载壳拓扑，使整库可移植备份还原。

```
        下载源 (SP1: 检测 + 管理各自的 env/config)
  HF   Ollama   ModelScope   URL   PyTorchHub   Civitai   Git
   \      \        |          |        /          /        /
    \------\-------+----------+-------/----------/--------/
                       aim download / ingest(SP2 摄取)
                               |
                               v
  ┌──────────────────────────────────────────────────────────┐
  │  ~/AI/store/{category}/{model_id}/   ← 唯一真实副本(扁平)   │
  └──────────────────────────────────────────────────────────┘
       ▲              ▲                 ▲               ▲
  (SP2 重建的"加载壳",全部指回 store)
  ┌────┴───────┐ ┌────┴─────────┐ ┌────┴────────┐ ┌───┴──────────┐
  │ HF_HOME/hub│ │ OLLAMA_MODELS│ │ MODELSCOPE_ │ │ 引擎目录      │
  │ models--…/ │ │  blobs/ +    │ │ CACHE/models│ │ (omlx/comfyui │
  │ snapshots ─┼→│  manifests/ ─┼→│ /org/repo ──┼→│  /whisper)───┼→ store
  └────────────┘ └──────────────┘ └─────────────┘ └──────────────┘

  registry.json: 每模型 storage{class, links[], reconstruct{...}} + 库级 manifest
```

**三层子项目（各自独立 spec → plan → 实现）**：

| | 范围 | 关键产出 |
|---|---|---|
| **SP1（本文）** | 源/工具模型 + 环境变量管理 | 统一源描述符；跨 shell/OS 检测与写入；扩展到 PyTorch Hub/Civitai/Git；配置对齐现实 |
| **SP2** | 原生摄取→store + 加载壳重建 + storage 标注 | 正确的 Q2 摄取（取代现有会翻倍/留 CAS 结构的 convert）；HF/Ollama/MS 加载壳；registry `storage` 块 |
| **SP3** | 可移植备份/还原 | `aim backup`/`aim restore`，消费 SP2 标注，跨机重建壳 + 重设 env |

**SP1 ↔ SP2 的安全耦合**：把 `HF_HOME` 等**缓存目录变量**直接改到 `~/AI` 下，只有在 SP2 能先把已存在的
`~/.cache` 模型摄取进 store 之后才安全（否则工具以为缓存空了、重新下载）。因此 **SP1 默认「采纳现状」**：
检测并让 aim 配置匹配真实位置，只主动管理安全变量（endpoint/token/accel/proxy）；缓存目录搬迁归 SP2。

---

## 1. SP1 目标与非目标

### 目标
1. **统一源描述符**：以一张静态表 `SOURCES` 描述每个源的下载工具、环境变量清单、缓存布局类型。
2. **检测引擎**：跨 shell/OS 求出每个相关环境变量的「有效值 + 来源 + 状态」，并把各源真实缓存位置写回 aim 配置
   （**即时修复「aim scan 找错位置」的 bug**）。
3. **写入器**：以「aim 独占 env 文件 + rc 仅加一行受管 source」的方式，跨 shell（zsh/bash/fish）跨 OS（macOS/Linux）
   管理安全环境变量；secret 走工具原生库或 600 文件；服务级变量走 launchctl/systemd。
4. **命令集**：`aim env [show|apply|path]`、`aim sources [list|install]`，并把检测接入 `aim scan`。
5. **范围扩展**：把 PyTorch Hub、Civitai、Git/Git-LFS 纳入源描述符（env/工具层），与既有 HF/Ollama/MS/URL 并列。

### 非目标（明确推迟）
- 缓存目录搬迁到 `~/AI` 下（含 `launchctl`/systemd 实际改服务路径）——属 SP2，需与字节迁移、加载壳重建原子完成。
- 原生 CAS → store 摄取、加载壳重建、registry `storage` 标注——属 SP2。
- 备份/还原——属 SP3。
- Windows 支持——v1 检测到即清晰报"暂不支持，请手动设置/`setx`"。
- 各源的实际「下载」逻辑改造——SP1 不改下载执行路径（仅扩展描述符与工具检测）。

---

## 2. 组件与边界（每个可独立测试）

| 组件 | 职责 | 输入 | 输出 / 副作用 |
|---|---|---|---|
| `SOURCES`（数据） | 静态描述每源的 tools/env/cache_layout | — | 供其余组件查询 |
| `EnvDetector`（读） | 求每变量有效值 + 来源 + 状态；解析每源缓存目录 | `SOURCES`、HOME、shell | 检测报告；写回 `config.sources.<k>.cache_path` |
| `ShellWriter`（写） | 生成 env 文件、接线 rc 受管块（跨 shell/OS、幂等、可 dry-run） | 待管理的非 secret 变量、目标 shell | `~/.aim/env.{sh,fish}`、rc 受管块、`<rc>.aim.bak` |
| `SecretStore`（写） | token 写入工具原生库或 600 文件，绝不入 env.sh/rc；输出脱敏 | secret 变量值 | `hf auth` 等 / `~/.aim/secrets.env` |
| `ServiceEnv`（写） | 守护进程级变量（仅检测+报告，SP1 不改） | 服务变量 | 报告 / 指令文本 |
| `aim env` / `aim sources`（CLI） | 编排上述组件，呈现/落盘 | argparse | 终端输出 / 配置与文件变更 |

---

## 3. 统一源描述符 `SOURCES`

取代现有 `_BACKEND_REGISTRY`（仅含 `tools`）。每个源：

```python
SOURCES["huggingface"] = {
  "aliases": ["hf"],
  "tools": [ {name, check, install_cmd, description}, ... ],   # 沿用现有 install/检测形状
  "cache_layout": "cas-hf",     # cas-hf | cas-ollama | flat-ms | torch-hub | flat   (SP2 消费)
  "env": [
    {"name":"HF_HOME","role":"cache_dir","default":"~/.cache/huggingface","subpath":"hub",
     "detect":["env","rc","tool"],"manage":"env_file","secret":False},
    {"name":"HF_HUB_CACHE","role":"cache_dir_override","default":"$HF_HOME/hub",
     "detect":["env","rc","tool"],"manage":"env_file","secret":False},
    {"name":"HF_ENDPOINT","role":"endpoint","default":"https://huggingface.co","manage":"env_file"},
    {"name":"HF_TOKEN","role":"token","secret":True,"manage":"native"},
    {"name":"HF_HUB_ENABLE_HF_TRANSFER","role":"accel","default":"0","manage":"env_file"},
    {"name":"HF_XET_CACHE","role":"regen_cache","default":"$HF_HOME/xet"},   # 可再生 → SP3 备份排除
  ],
}
```

`role` 取值：`cache_dir` | `cache_dir_override` | `endpoint` | `token` | `accel` | `proxy` | `offline` | `regen_cache` | `misc`
`manage` 取值：`env_file`（写 aim env 文件）| `native`（工具原生命令/库）| `service`（守护进程级）| `none`（仅检测）

### 3.1 各源 env 清单（实现参照）

| 源 | cache 变量(默认 / 子目录) | endpoint | token(secret) | accel/其他 | 可再生(SP3 排除) | cache_layout |
|---|---|---|---|---|---|---|
| huggingface | `HF_HOME`(`~/.cache/huggingface` / `hub`)；`HF_HUB_CACHE` 覆盖 | `HF_ENDPOINT` | `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`→`hf auth` | `HF_HUB_ENABLE_HF_TRANSFER`、`HF_HUB_OFFLINE` | `HF_XET_CACHE`、logs | cas-hf |
| ollama | `OLLAMA_MODELS`(`~/.ollama/models`，`blobs/`+`manifests/`) | `OLLAMA_HOST` | — | 服务级（不读 shell rc） | — | cas-ollama |
| modelscope | `MODELSCOPE_CACHE`(`~/.cache/modelscope`，`models/`+`hub/`) | 域名/endpoint 变量 | token 变量（`modelscope login`）→见 §9 待核实 | — | `.lock/`、`._____temp/` | flat-ms |
| pytorch-hub | `TORCH_HOME`(`~/.cache/torch` / `hub`) | — | — | 受 `XDG_CACHE_HOME` 影响 | — | torch-hub |
| civitai | 无（aim 直接落 store） | API base | `CIVITAI_API_TOKEN`→见 §9 待核实 | — | — | flat |
| git/git-lfs | 无（clone 进 store） | `http.proxy`(git config) | git credential helper | `GIT_LFS_SKIP_SMUDGE` | — | flat |
| url | 无 | — | — | `HTTP(S)_PROXY`（已支持） | — | flat |

---

## 4. 检测引擎 `EnvDetector`（纯读）

对每个变量按优先级求「有效值」并记录来源：

1. **登录 shell 回显**（捕获 rc 里定义的）：按检测到的 shell 跑
   `zsh -ic 'printf %s "$HF_HOME"'` / `bash -lic '…'` / `fish -c 'echo $HF_HOME'`
2. **rc 文件扫描**：解析 `export NAME=` / `set -gx NAME` / 纯赋值，记录在哪个文件定义、是否多处冲突
3. **工具自报**：如 `python3 -c "import torch.hub;print(torch.hub.get_dir())"`、`hf env`、`git config --get http.proxy`
4. **描述符默认值** 兜底

每项输出：`{name, effective_value, source: env|rc:<file>|tool|default, aim_recommended, status: ok|drift|unset|conflict}`
（`conflict` = 多个 rc 文件给出不同值）。

**缓存目录解析**：每源的有效缓存位置 = 解析其 `cache_dir` 变量，并尊重覆盖关系（如 `HF_HUB_CACHE` > `HF_HOME/hub`）。

**配置对齐现实（修 scan bug）**：把解析出的真实缓存位置写回 `config.json` 的 `sources.<key>.cache_path`，
让 `aim scan` 去对的地方找模型。这是 SP1 的即时价值——不动任何用户文件就先让 aim 看见 `~/.cache` 里的现有模型。

---

## 5. 写入器 `ShellWriter`（跨 shell/OS、幂等）

- **shell 检测**：`$SHELL` 为主 + 探测存在哪些 rc；`--shell zsh|bash|fish|all` 覆盖，默认登录 shell。
- **OS 检测**：`platform.system()` → darwin/linux 一等公民；windows v1 不支持（aim 依赖 symlink/hardlink），报清晰提示。
- **env 文件**（每次 `apply` 整体重写 → 幂等）：
  - `~/.aim/env.sh`（posix）：`export NAME="value"`，按源分区带注释
  - `~/.aim/env.fish`：同内容，`set -gx NAME "value"`
  - 仅写 `manage:env_file` 且非 secret 项；**cache 目录变量在 SP1 默认不写**（采纳现状）
- **rc 接线**——只插一个受管块（含唯一一行 source）：
  ```sh
  # >>> aim env >>>
  [ -f "$HOME/.aim/env.sh" ] && . "$HOME/.aim/env.sh"
  # <<< aim env <<<
  ```
  fish 版：`test -f "$HOME/.aim/env.fish"; and source "$HOME/.aim/env.fish"`
  - 目标：zsh→`~/.zshrc`；bash→`~/.bashrc`（并确保 `~/.bash_profile`/`~/.profile` 会载入 `~/.bashrc`）；
    fish→`~/.config/fish/config.fish`（按需建目录）；未知→`~/.profile`
  - 幂等：块存在则替换、否则追加；首次改动前备份 `<rc>.aim.bak`；`--dry-run` 只打印 diff、不写

---

## 6. Secret 处理 `SecretStore`（token 绝不进 env.sh/rc）

1. **优先工具原生库**：HF→`hf auth login --token`（写 `$HF_HOME/token`，600）；MS→`modelscope login`；git→credential helper
2. **无原生库的兜底**（Civitai）：`~/.aim/secrets.env`（chmod 600），由 env.sh 用
   `[ -f "$HOME/.aim/secrets.env" ] && . "$HOME/.aim/secrets.env"` 载入；默认排除于备份/分享
3. OS keychain（macOS `security` / Linux `secret-tool`）留作将来，非 v1
4. `aim env` 输出对 secret 只显示 `set`/`unset`，不打印值

---

## 7. 服务级 env `ServiceEnv`（守护进程不读 shell rc）

仅 `OLLAMA_MODELS`/`OLLAMA_HOST`（ollama server）：
- macOS：`launchctl setenv OLLAMA_MODELS <path>`（提示菜单栏 app 需重启）
- Linux：systemd drop-in `~/.config/systemd/user/ollama.service.d/aim.conf`（`Environment=…`）→ `daemon-reload`+restart
- **SP1 只检测并报告**服务有效值；真正改动在 SP2 搬迁时用 `--service` 触发。

---

## 8. 命令集与配置

### 8.1 命令
| 命令 | 作用 |
|---|---|
| `aim env [show] [--json]` | 检测 + 表格（每源：变量→当前/来源/推荐/状态，secret 脱敏）+ 每源解析出的 cache 目录。只读 |
| `aim env apply [--shell zsh\|bash\|fish\|all] [--service] [--dry-run]` | 写 env 文件 + rc source 行（+ 服务 env）；备份 rc；SP1 默认只写安全变量(endpoint/accel) + 接线 |
| `aim env path <source>` | 打印某源解析后的 cache 目录（脚本用） |
| `aim sources [list] [--json]` | 列出已知源 + 工具安装状态 + env 概览 |
| `aim sources install <key>` | 复用已合并的 `_ensure_backend` 安装该源的下载工具 |

并把检测接入 `aim scan`（自动对齐 `cache_path`）+ 轻量 `aim env --check`（漂移/冲突）。

### 8.2 config.json 增量
```json
"sources": {
  "huggingface": {"cache_path":"/Users/hjma/.cache/huggingface","relocated":false,
                  "managed_env":{"HF_ENDPOINT":"https://hf-mirror.com","HF_HUB_ENABLE_HF_TRANSFER":"1"}},
  "pytorch-hub": {"cache_path":"~/.cache/torch","relocated":false}
},
"env": {"shells":["zsh"],"files":{"posix":"~/.aim/env.sh","fish":"~/.aim/env.fish"},"managed":true,
        "rc_backups":["~/.zshrc.aim.bak"]}
```

### 8.3 目标性清理（serves the goal，不做无关重构）
让 `sources.<key>.cache_path` 成为缓存位置的**唯一权威**；HF/ollama/MS 的 scan 适配器改读它，
而非现在硬编码的 `engines.<key>.model_dir` 相对路径（这正是「aim 找错位置」的根因）。`engines` 保留给 provision/link 侧。
迁移时为旧 config 提供向后兼容默认（无 `sources` 段时由 `engines` + 检测推导生成）。

---

## 9. 待实现期核实项（非占位，是带默认值的明确核实任务）
- **ModelScope token 变量名**：当前最佳已知 `MODELSCOPE_API_TOKEN` / `modelscope login --token`；实现时以 `modelscope --help` 与源码核实。
- **ModelScope endpoint/域名变量名**：实现时核实（用于镜像/自建）。
- **Civitai token 变量名**：当前最佳已知 `CIVITAI_API_TOKEN`（社区约定）；实现时以所选下载方式（API / civitdl）核实。
- **HF token 文件路径**：`$HF_HOME/token`；以 `huggingface_hub.constants` 核实。

---

## 10. 测试策略（贴合 aim「零依赖、stdlib」风格）
- **纯单测**（不需真实工具）：
  - `EnvDetector`：喂假 rc 文件 + 打桩 login-shell 回显 → 断言有效值/来源/drift/conflict/unset；缓存解析尊重覆盖关系
  - `ShellWriter`：临时 HOME → 断言 env.sh/env.fish 内容、受管块插入/替换**幂等**（apply 两次结果相同）、rc 备份、`--dry-run` 不写
  - `SecretStore`：600 权限、不出现在 env.sh/rc、`aim env` 脱敏
  - `SOURCES` 描述符 schema 合法性（每 env 项含必需键；`role`/`manage`/`cache_layout` 在允许集合内）
  - config 往返 + 旧 config 向后兼容
- **轻量 e2e**（工具缺失则 skip，沿用现有 e2e skip 模式）：`hf`/`git`/`python+torch` 在则探真实值
- **跨 shell**：对 zsh/bash/fish 目标文件参数化（纯字符串生成，无需真跑 shell）

---

## 11. 验收标准
1. 在「模型在 `~/.cache` 默认位置、无任何 env 配置」的机器上，`aim env show` 正确报出各源真实缓存位置与状态；
   `aim scan` 随后能扫到这些模型（配置自动对齐现实），全程不修改用户任何 shell 文件。
2. `aim env apply` 在 zsh/bash/fish 下分别正确生成 env 文件并接线 rc 受管块；重复执行幂等；改动前有备份；`--dry-run` 不落盘。
3. token 不出现在任何 env 文件 / rc / 备份中；`aim env` 输出对 secret 脱敏。
4. PyTorch Hub / Civitai / Git 进入 `SOURCES`，`aim sources list` 能列出其工具安装状态与 env 概览。
5. 全部单测通过（`make test`），不引入第三方依赖。
