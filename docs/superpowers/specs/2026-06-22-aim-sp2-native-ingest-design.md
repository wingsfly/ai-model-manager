# SP2 — 原生摇入 → store + 加载壳重建 + storage 标注（设计 spec）

- 日期：2026-06-22
- 状态：已批准（待写实现计划）
- 分支：`feat/aim-sp2-native-ingest`
- 依赖：SP1（已合并到 main）—— `SOURCES` 描述符、`EnvDetector`、`sources.<key>.cache_path` 检测对齐、HF/Ollama scan 读 cache_path
- 作者：wingsfly + Claude

---

## 0. 背景与定位

承接整体架构（store 为唯一真源 + 各工具经"重建的原生加载壳"指回 store + 可移植 manifest 标注，详见 SP1 spec §0）。
SP2 实现这套架构的核心：把 HF / Ollama / ModelScope 三种"自带 CAS/缓存格式"的模型**正确摇入** `store/{category}/{id}/` 扁平单目录，
在工具原缓存位置**重建加载壳**指回 store，并写入 registry 的 `storage` 标注，使工具仍原生加载、且 SP3 能可移植备份还原。

SP2 同时**取代现有坏掉的 `aim convert`**（`op_convert_native_to_store` 用 `shutil.copytree` 解引用 → 体积翻倍、复制出 CAS 结构、不建壳，工具无法按 repo-id 加载）。

### 已确认的关键决策（2026-06-22 与用户）
1. **壳位置 = 原地，不搬迁缓存**：在工具当前缓存位置（`HF_HOME`/`OLLAMA_MODELS`/`MODELSCOPE_CACHE` 现在指向处）重建壳；不改任何环境变量。备份只需 `store/` + manifest（壳可再生）。
2. **Ollama**：GGUF 提取进 store，ollama `blobs/sha256-*` 硬链回 store 的 GGUF；manifest+digests 记入标注。与 HF/MS 一致"store 单副本"。
3. **跨源去重**：同模型在 HF/MS 都有时，**每源各一条 store 记录**（各自 shim/provenance），再复用现有 `aim dedup`（大小+哈希→硬链）把相同文件合成同 inode。逻辑两条、物理一份。

### 非目标
- 缓存搬迁到 `~/AI`（SP1 已实现 `aim env apply --relocate` 的接线；SP2 不强制搬迁）。
- 备份/还原命令（`aim backup`/`restore`）—— SP3。SP2 只**产出**可被 SP3 消费的 `storage` 标注，并通过"往返测试"+`verify --fix` 预验证重建逻辑。
- 非 CAS 引擎（omlx/comfyui/whisper/coqui/…）的摇入：它们已有 provision（软/硬链回 store）、本就按路径加载，不在范围。

---

## 1. 组件与边界（各自可独立测试）

| 组件 | 职责 | 依赖 |
|---|---|---|
| `NativeReader`（HF/Ollama/MS 各一） | 读出原生模型的**真实文件清单**及重建所需元数据（HF: 解析 `snapshots/<commit>/` 软链到真实 blob + 取 commit；Ollama: gguf blob + 原始 manifest + 小 blob；MS: 扁平目录文件 + `.msc/.mdl/.mv`） | `SOURCES`、cache 路径 |
| `StoreIngestor` | 把真实文件复制成 `store/{category}/{id}/` 扁平单目录（HF deref 每文件只复制一份 → 无 2×）；校验；写 registry `storage` 标注 | `NativeReader`、`Registry` |
| `ShimBuilder`（HF/Ollama/MS 各一） | 在工具当前缓存位置重建加载壳指回 store | `storage` 标注 |
| `ModelScopeAdapter`（新） | 补 MS 的 `scan`（`flat-ms` 布局）+ `provision`（返回 `[]`），使 MS 模型能被发现/纳管 | `EngineAdapter` |
| `op_ingest` / 改造 `op_convert` | 命令入口，编排 reader→ingestor→shimbuilder→annotate；`--dry-run`/可回滚 | 全部 |
| `op_verify` 扩展 | 校验/重建 `storage` 壳（`--fix` 从标注重建） | `ShimBuilder` |

所有产线代码进单文件 `aim.py`（保持零依赖、`ln -s aim.py` 安装模型）。

---

## 2. storage 标注 schema（registry 每模型新增 `storage` 块）

**可移植**（相对路径 + `cache_root_var` 在目标机重算绝对路径）且**可重建**（commit/digests/manifest/元数据）。这是 req2 的"标注信息"，也是 SP3 的输入。

```jsonc
"storage": {
  "class": "managed-hf | managed-ollama | managed-ms | managed-flat",   // 备份还原据此分流
  "store_path": "store/asr/model/hf-firered-aed-l",                     // 相对 root
  "ingested_at": "2026-06-22T...Z",
  "shims": [ <shim>, ... ]                                              // 通常一条(每记录=一个工具+一份 store 副本)。跨源同模型是两条独立记录(§0.3)，由 aim dedup 在文件层硬链去重
}
```

每种 `shim`：

**HF（`kind: hf-cas`）**
```jsonc
{ "tool": "huggingface", "kind": "hf-cas",
  "location": "huggingface/hub/models--FireRedTeam--FireRedASR-AED-L",  // 相对 HF_HOME
  "cache_root_var": "HF_HOME",
  "reconstruct": { "repo_id": "FireRedTeam/FireRedASR-AED-L", "commit": "e57f59...",
                   "files": ["config.yaml", "model.pth.tar", ...] } }   // snapshot 文件 → 软链 store
```
**Ollama（`kind: ollama-cas`）**
```jsonc
{ "tool": "ollama", "kind": "ollama-cas",
  "location": "manifests/registry.ollama.ai/library/<model>/<tag>",     // 相对 OLLAMA_MODELS
  "cache_root_var": "OLLAMA_MODELS",
  "reconstruct": { "model": "<model>", "tag": "<tag>", "manifest": { <raw manifest JSON> },
                   "blobs": [ {"digest": "sha256-...", "store_file": "<id>.gguf", "kind": "hardlink"},
                              {"digest": "sha256-...", "store_file": "blob-sha256-...", "kind": "copy"} ] } }
```
**ModelScope（`kind: ms-dir`）**
```jsonc
{ "tool": "modelscope", "kind": "ms-dir",
  "location": "models/Qwen/Qwen3-ASR-0___6B",                           // 相对 MODELSCOPE_CACHE；保留原始(可能改名)目录名
  "cache_root_var": "MODELSCOPE_CACHE",
  "reconstruct": { "repo_id": "Qwen/Qwen3-ASR-0.6B", "layout": "models|hub-models",
                   "metadata_files": [".msc", ".mdl", ".mv"] } }        // 元数据已在 store；壳=目录软链
```

---

## 3. 通用摇入流程（每模型，安全可回滚）

1. **读**：`NativeReader` 列出真实文件 + 重建元数据。
2. **复制**：→ `store/{category}/{id}/` 扁平单目录（**copy 优先**；HF 每文件只复制一份 → 无 2×；ollama gguf→`{id}.gguf`、小 blob 与 manifest.json 一并入 store；MS 含 `.msc/.mdl/.mv`）。校验文件数+大小。
3. **建壳**：`ShimBuilder` 在原缓存位置重建加载壳指回 store（见 §4），并校验壳能解析到 store。
4. **标注**：写 registry `storage` 块，`native_cas=False`，更新 category/format/size。
5. **清理**：壳校验通过后删除原生冗余字节；任一步失败 → 回滚（用 store 副本还原原生 / 删半成品 store / 不写标注）。
- `--keep-native`：保留原生字节（暂时 2×、最稳）；默认删原生回收空间。
- `--dry-run`：只预览；批量前 registry 自动 `.bak`（现有机制）。
- 每模型独立事务：模型 N 失败不影响 N-1。

---

## 4. 三种壳重建细节

### HF（`{HF_HOME}/hub/models--{org}--{repo}/`）
```
refs/main                       ← 写入 commit hash（来自原缓存 refs/main）
snapshots/<commit>/<file>       ← 绝对软链 → store/{cat}/{id}/<file>（逐文件）
(blobs/ 删除：HF 按 snapshots 读取，不在本地加载时校验 blob 结构)
```
`from_pretrained("org/repo")` 走 refs→snapshot→软链→store 原生加载。软链用**绝对路径**（壳由标注在目标机重建，不依赖软链可移植）。

### Ollama（`{OLLAMA_MODELS}/`）
store 目录含 `{id}.gguf` + `manifest.json` + 小 blob 文件。壳：
```
blobs/sha256-<gguf_digest>      ← 硬链 → store 的 {id}.gguf（同卷；同 inode → digest 不变；跨卷则 copy）
blobs/sha256-<small_digest>     ← 小 blob 从 store 复制回
manifests/<registry>/<ns>/<model>/<tag>  ← 写回 manifest.json
```
`ollama run` 照常（按 manifest→digest→blob 读，内容一致）。

### ModelScope（`{MODELSCOPE_CACHE}/{layout}/{org}/{repo}`）
MS 模型是扁平真实文件 + `.msc/.mdl/.mv`（摇入时一并入 store）。壳 = 把缓存里的模型目录换成**目录软链** → `store/{cat}/{id}/`。
处理 MS 的目录改名（`0.6B`→`0___6B`）与两种布局（`models/` vs `hub/models/`）：标注记录原始目录名与 `layout`。

### ModelScopeAdapter（新）
- `name="modelscope"`、`native_cas=True`、formats `safetensors/bin/pt/gguf`；注册进 `ADAPTERS`/`ENGINE_NAMES`/`default_config.engines`。
- `base_path` 读 SP1 同步的 `sources.modelscope.cache_path`；`scan()` 遍历 `{cache}/models/{org}/{repo}` 与 `{cache}/hub/models/{org}/{repo}`，跳过 `._____temp`/`.lock`/隐藏；每个含真实文件的 repo 目录 → `ScannedModel(native_cas=True, is_directory=True, source={type:modelscope,repo_id})`。
- `provision()` 返回 `[]`（与 HF/ollama 一致；壳由 `ShimBuilder` 在摇入时建）。

---

## 5. 命令面

| 命令 | 作用 |
|---|---|
| `aim ingest <model_id> [--dry-run] [--keep-native] [--new-id ID] [--category CAT]` | 摇入单个 native-CAS 模型 |
| `aim ingest --all-native [--dry-run] [--keep-native]` | 批量摇入所有 `native_cas=True` 模型 |
| `aim convert …`（改造） | **委托给 ingest**（保留命令名 + `--new-id/--category`，加弃用提示）；旧的坏实现彻底取代 |
| `aim verify [--fix]`（扩展） | 额外校验 `storage` 壳；`--fix` 从 `storage.reconstruct` 重建坏壳 |

摇入**显式触发**，不混入 `scan`（scan 保持只读/登记）。

---

## 6. 安全/回滚（细化）
- **copy 优先**（非 move）：原生字节在壳校验通过前始终完好；失败直接弃置 store 半成品。
- 壳校验：HF 检查每个 snapshot 软链 `resolve()` 落在 store 内且目标存在；ollama 检查 blob 与 store gguf `st_ino` 相同 + manifest 文件存在；MS 检查目录软链指向 store 且 `.msc` 等存在。
- 回滚点：复制失败→删 store 半成品；建壳失败→删 store 半成品 + 还原原缓存目录（copy 模式下原生未动，无需还原）；标注写入失败→删壳+store、还原原生。
- 默认删原生回收空间需谨慎：仅在壳校验通过后；`--keep-native` 跳过删除。

---

## 7. 测试策略（stdlib unittest；合成原生缓存，不需真实工具）
- **摇入正确性（每工具）**：合成 HF（`models--/blobs/snapshots` 含软链）、ollama（`blobs/sha256-*`+`manifests`）、MS（扁平+`.msc`）缓存 → 摇入 → 断言：store 为扁平真实文件、**store 总大小 == 原真实大小（无 2×）**、壳解析回 store、`storage.reconstruct` 内容正确。
- **加载仍可用**：HF 断言 `snapshots/<commit>/<file>` 读到的内容 == store 文件内容；ollama 断言 blob 与 store gguf **同 inode** + manifest 存在且引用正确 digest；MS 断言目录软链→store + 元数据齐全。
- **往返（预验 SP3）**：摇入 → 删除壳 → 仅凭 `storage.reconstruct` 调 `ShimBuilder` 重建 → 断言壳解析到 store。
- **回滚**：注入建壳失败（如目标不可写/打桩抛错）→ 断言原生完好、无 `storage` 标注、store 半成品已清理。
- **ModelScopeAdapter**：合成两种布局的 MS 缓存 → `scan` 命中所有 repo、跳过 `._____temp`/`.lock`；改名目录（`0___6B`）正确解析 repo_id。
- **dedup 交互**：同模型两源各摇入一条 → `aim dedup --apply` 后两条 store 记录的相同文件 `st_ino` 相同。
- **convert 委托**：`aim convert <native-id>` 走新 ingest 路径（不再 2×、建了壳），并打印弃用提示。
- **e2e**：真实 `hf`/`ollama`/`modelscope` 存在则跑（否则 skip，沿用现有 e2e skip 模式）。

---

## 8. 验收标准
1. 对合成的 HF/Ollama/MS 缓存，`aim ingest` 后：store 为扁平单目录、**无体积翻倍**、原工具仍能原生加载（壳解析正确），registry 有完整 `storage` 标注。
2. `aim convert <id>` 不再产生 2× 的 CAS 副本，而是走正确摇入（建壳 + 标注）。
3. `ModelScopeAdapter` 能 `scan` 出你 `~/.cache/modelscope` 里的 MS 模型（两种布局、改名目录）。
4. `aim verify --fix` 能从标注重建被删/损坏的壳（往返测试通过）。
5. 摇入失败可回滚，原生字节不丢；`--dry-run` 不改任何文件。
6. 同模型两源摇入后 `aim dedup` 能把相同文件硬链为同 inode。
7. 全部单测通过（`make test`），无第三方依赖。
