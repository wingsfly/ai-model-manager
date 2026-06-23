# SP4 — flat-file 摄取：PyTorch Hub + Whisper 缓存（设计 spec）

- 日期：2026-06-23
- 状态：已批准（待写实现计划）
- 分支：`feat/aim-sp4-flatfile-ingest`
- 依赖：SP1（SOURCES/EnvDetector,含 OLLAMA_MODELS 服务检测)、SP2(storage 标注 + op_ingest + verify --fix + 各 build_shim)、SP3(backup/restore + _retarget_shim_locations)——均在 main
- 作者：wingsfly + Claude

---

## 0. 背景与定位

承接整体架构(store 为唯一真源 + 工具经重建壳加载 + 可移植 manifest)。SP2 覆盖了 CAS(hf/ollama)与目录(ms)两类原生格式;
SP4 补上第三类:**单权重文件模型**(一个 `.pth`/`.pt` 即一个模型),并据此把两个真实存在的外置下载源纳入摄取:
- **PyTorch Hub**(`$TORCH_HOME/hub/checkpoints/*.pth|*.pt`,用户机有 wav2vec2 360M)。
- **openai-whisper 下载缓存**(`${XDG_CACHE_HOME:-~/.cache}/whisper/*.pt`,用户机有 base.pt 139M)。

经盘点(在用户 MacBook Pro 上实测),其余候选要么已被覆盖(Civitai→ComfyUI 引擎;MLX/diffusers/transformers/vLLM→HF 缓存;
git→下载进 store),要么未安装(LM Studio/Jan/GPT4All/Kaggle/TF Hub)。Vosk/自训练模型无下载工具约定,更适合 `aim import`,不在本期。

### 已确认决策(2026-06-23 与用户)
1. **PyTorch Hub 仅摄取 `checkpoints/*.pth|*.pt`**(纯权重);`<owner>_<repo>_<ref>/` 代码仓不纳管(主要是代码、内嵌权重很小、torch.hub 可秒级重新 clone)。
2. **whisper 缓存并入本期**,作为独立源 `whisper-cache`,与现有 `whisper` 引擎(管 `~/AI/whisper` 的 provision)区分。

### 非目标
- torch.hub 代码仓的摄取(决策①排除);Vosk / 自训练模型(用 `aim import`);LM Studio/Jan/GPT4All/Kaggle/TF Hub(未安装)。

---

## 1. 新增摄取类型 `flat-file`

一个模型 = 单个权重文件(非目录、非 CAS)。
- **摄取**:复制该文件 → `store/{category}/{id}/{文件名}`(扁平单目录,内含一个文件)。
- **壳**:在原缓存位置留**文件软链** → store 里的文件(工具按原路径加载,照常)。
- **标注**(registry `storage`):
```jsonc
{ "class": "managed-torch | managed-whisper", "store_path": "store/<cat>/<id>",
  "ingested_at": "...",
  "shims": [{ "tool": "pytorch-hub | whisper-cache", "kind": "flat-file",
              "location": "<绝对或 root 相对的原文件路径>", "cache_root_var": "TORCH_HOME|XDG_CACHE_HOME",
              "reconstruct": { "filename": "<file>", "rel": "<相对该源 cache_dir 的子路径>" } }] }
```
  - torch:`rel = "checkpoints/<file>"`(cache_dir=`$TORCH_HOME/hub`);whisper:`rel = "<file>"`(cache_dir=`${XDG_CACHE_HOME:-~/.cache}/whisper`)。

---

## 2. 组件与边界(各自可测)

| 组件 | 职责 |
|---|---|
| `_flatfile_read_native(file_path)` | 返回 `{files:[{name, real_path, size}]}`(单文件);`name`=文件名 |
| `_flatfile_build_shim(orig_path, store_file)` | 在 `orig_path` 建**文件软链** → `store_file`;**改名旁置→建链→失败还原**(无数据丢失窗口,同 MS) |
| `op_ingest` 的 `flat-file` 分支 | 按源的 `cache_layout=="flat-file"` 走:copy→store → build_shim → 标注;copy 优先、失败回滚 |
| `PyTorchHubAdapter`(scan) | 扫 `$TORCH_HOME/hub/checkpoints/*.pth|*.pt` → `native_cas` 单文件模型;忽略代码仓 + `trusted_list` |
| `WhisperCacheAdapter`(scan) | 扫 `${XDG_CACHE_HOME:-~/.cache}/whisper/*.pt` → `native_cas` 单文件模型 |
| `_retarget_shim_locations`(SP3,扩展) | `flat-file`:`location = cache_dir(shim["tool"]) / reconstruct["rel"]` |
| `_rebuild_shim_from_storage`(SP2,扩展) | `flat-file`:重建文件软链 `location → store/{…}/{file}` |

复用:`_ingest_to_store(info["files"], store_dir)`(SP2,按 files 列表复制进目录——单文件列表同样适用)。SP3 backup/restore、SP2 verify、dedup 无需改动(flat-file 模型在 store 内,自动被带走/重建/去重)。

全部产线代码进单文件 `aim.py`(归入既有 `# ── Native Ingest (SP2) ──` 段;新 adapter 放在其它 adapter 旁、`ADAPTERS` 之前)。

---

## 3. SOURCES / 注册

- `pytorch-hub`:SP1 已有(`TORCH_HOME` role=cache_dir subpath=`hub`,cache_layout=`torch-hub`)。本期把其 `cache_layout` 视为 `flat-file`(或在 op_ingest 中按 source key 判定走 flat-file 分支);新增 `PyTorchHubAdapter` 并注册。
- 新增 `whisper-cache`:
```python
"whisper-cache": {
  "aliases": ["whisper-dl"],
  "cache_layout": "flat-file",
  "tools": [{"name": "whisper", "check": "which",
             "install_cmd": "pip3 install --break-system-packages -U openai-whisper",
             "description": "openai-whisper"}],
  "env": [{"name": "XDG_CACHE_HOME", "role": "cache_dir", "default": "~/.cache",
           "subpath": "whisper", "detect": ["env", "rc"], "manage": "none", "secret": False}],
}
```
  → `cache_dir("whisper-cache")` = `${XDG_CACHE_HOME:-~/.cache}/whisper`(openai-whisper 真实约定)。
- `PyTorchHubAdapter.name="pytorch-hub"`、`WhisperCacheAdapter.name="whisper-cache"`,均 `native_cas=True`,注册进 `ADAPTERS` + `ENGINE_NAMES` + `default_config()["engines"]`;`base_path` 读 `sources["<name>"]["cache_path"]`(SP1 reality-sync 已会写入,torch 经其 TORCH_HOME 检测、whisper 经 XDG 检测)。

> 注:`pytorch-hub` 的 `cache_layout` 在 SP1 设为 `torch-hub`;本期统一改为 `flat-file`(语义即"单文件摄取"),并更新 SP1 测试中对该值的断言(若有)。

---

## 4. 摄取流程(flat-file,安全可回滚)

1. 选中模型(`native_cas=True`,来自 `pytorch-hub`/`whisper-cache` 扫描;`canonical.path` = 原文件路径)。
2. `_flatfile_read_native(cache_file)` → 单文件清单。
3. `store_dir = root.store_path / category / id`;`_ingest_to_store(files, store_dir)`(复制单文件;copy 优先)。
4. `_flatfile_build_shim(cache_file, store_dir/filename)`:把 `cache_file` 改名旁置(`.aim-old`)→ 建软链 → 成功删旁置 / 失败把旁置改回(原文件不丢)。
5. 写 `storage` 标注(class=`managed-torch`/`managed-whisper`,shim kind=`flat-file`,reconstruct 见 §1),`native_cas=False`,更新 category/format/size。
6. 任一步失败 → 回滚:删半成品 store 目录;`_flatfile_build_shim` 已保证原文件还原。`--dry-run` 只预览。
   **关于 `--keep-native`**:flat-file 摄取后原路径变成软链(始终只有一份真实字节,在 store),因此 `--keep-native` 对 flat-file **无意义/被忽略**(无法在同一路径既留原文件又留软链)。

> category 推断:`PyTorchHubAdapter` 按文件名(`wav2vec2/w2v`→`asr/model`,`vad`→`audio/vad`,否则 `llm/chat` 兜底);`WhisperCacheAdapter` 一律 `asr/model`。

---

## 5. 测试策略(stdlib unittest;全合成,无需真实工具)
- `_flatfile_read_native`:单文件 → 清单(name/real_path/size)。
- `_flatfile_build_shim`:建文件软链解析到 store;**数据丢失安全**(打桩 `os.symlink` 抛错 → 原文件还原、无残留 `.aim-old`)。
- `PyTorchHubAdapter.scan`:合成 `hub/checkpoints/{wav2vec2…pth, x.pt}` + 一个代码仓目录 + `trusted_list` → 只命中 checkpoints 文件、类别推断正确、忽略目录/trusted_list。
- `WhisperCacheAdapter.scan`:合成 `whisper/{base.pt, large-v3.pt}` → 命中、类别 `asr/model`。
- `op_ingest` flat-file 端到端(torch + whisper 各一):store 有真实文件、原处是软链→store 且可读、标注 class/shim/reconstruct 正确、原生大小不翻倍。
- `_retarget_shim_locations` flat-file:注入 `EnvDetector`(给定 TORCH_HOME / XDG_CACHE_HOME)→ 断言 location 重算为 `cache_dir(tool)/rel`。
- `verify --fix` flat-file:删软链 → 从 store 重建。
- **异机往返**:torch checkpoint 源机 ingest → `op_backup` → 目标机(不同 TORCH_HOME,经注入 detector)`op_restore`(`registry_save=False`)→ 文件软链建在目标 `$TORCH_HOME/hub/checkpoints/<file>` 且解析到目标 store。
- `cache_dir("whisper-cache")`=`${XDG}/whisper`、`cache_dir("pytorch-hub")`=`$TORCH_HOME/hub` 各一断言。
- e2e:真实 torch/whisper 在则可跑(否则 skip)。

---

## 6. 验收标准
1. `aim scan` 能发现 `$TORCH_HOME/hub/checkpoints/*.pth` 与 `${XDG}/whisper/*.pt` 为 `native_cas` 单文件模型;代码仓被忽略。
2. `aim ingest <id>` 把单文件摄取进 `store/{cat}/{id}/`,原处留文件软链且工具仍按原路径加载;无体积翻倍。
3. 摄取失败可回滚,原文件绝不丢(`_flatfile_build_shim` 改名旁置+还原)。
4. `aim verify --fix` 能从标注重建被删的 flat-file 软链;`aim restore` 在不同 `TORCH_HOME` 的目标机正确重建 flat-file 软链到目标 store。
5. `whisper-cache` 源经 `${XDG_CACHE_HOME:-~/.cache}/whisper` 解析;`pytorch-hub` 经 `$TORCH_HOME/hub` 解析。
6. 全部单测通过(`make test`),无第三方依赖;现有 SP1/2/3 测试不回归。
