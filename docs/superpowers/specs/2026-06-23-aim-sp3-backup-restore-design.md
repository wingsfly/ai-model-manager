# SP3 — 可移植备份 / 还原（设计 spec）

- 日期：2026-06-23
- 状态：已批准（待写实现计划）
- 分支：`feat/aim-sp3-backup-restore`
- 依赖：SP1（env 检测/管理）、SP2（`storage` 标注 + `_rebuild_shim_from_storage` + 各 `_*_build_shim`）——均已合并到 main
- 作者：wingsfly + Claude

---

## 0. 背景与定位

整体架构第三层（store 为唯一真源 + 工具经重建壳加载 + 可移植 manifest）。SP3 实现「整库可移植备份/还原」：
把 `store/` 真实字节 + 一份自包含清单备份到外置盘/另一路径，并能在**换机/换盘、缓存路径不同**的目标机上一键还原——
重建 store、导入 registry、**按目标机重算各工具加载壳的位置**并重建壳、检测目标机 env 并提示。

SP3 不引入新的存储概念,而是**消费 SP2 的 `storage` 标注**(class/store_path/shims[location,cache_root_var,reconstruct])
并复用 SP1 的 `EnvDetector`/`op_env_apply` 与 SP2 的 `_rebuild_shim_from_storage`/`_*_build_shim`。

### 已确认的关键决策（2026-06-23 与用户）
1. **备份形态 = 目录 + 幂等同步**：`aim backup <dir>` 复制 `store/` 到 `<dir>/store/`(跳过同大小文件、可重跑/增量),写 `<dir>/aim-backup.json` 清单。对几百 GB 库友好、rsync 友好、无需解压。
2. **还原范围 = 重建 store+registry+壳(按目标机重算),env 只检测+提示**:`aim restore` 重建 store、导入 registry、为目标机重算并重建所有壳;检测并打印建议 env,但**不写 shell 配置**(`--apply-env` 才写)。与 SP1「env 写入显式化」一致、非侵入。

### 非目标
- 单文件压缩包备份(`--archive`)、增量快照历史(Time Machine 式)——非 v1。
- 选择性单模型备份/还原——v1 以整库为核心(`--model` 留作后续)。
- 备份加密、远程/云目标——非 v1(目标是本地目录/外置盘)。

---

## 1. 组件与边界（各自可独立测试）

| 组件 | 职责 |
|---|---|
| `_sync_store_dir(src, dst, verify=False)` | 幂等逐文件复制:目标已存在且同大小(或 `verify` 时同 `quick_hash`)则跳过;返回 (copied, skipped) 计数 |
| `_write_backup_manifest(config, registry, dst, store_files)` | 写 `<dir>/aim-backup.json`(models + sources + env + 完整性表) |
| `op_backup(config, registry, dest, verify, json_output)` | 编排:校验未摇入告警 → 同步 store → 写清单 |
| `_read_backup_manifest(dir)` | 读 + 校验 `aim-backup.json` 版本 |
| `_retarget_shim_locations(entry, detector)` | 用目标机缓存根 + `reconstruct` 身份**重算** `shims[].location`(HF/MS/ollama) |
| `op_restore(config, registry, src, root_id, apply_env, verify, json_output)` | 编排:读清单 → 重建 store → 导入 registry → 重算+重建壳 → 恢复源配置 → env 检测/提示 → verify |

复用:SP2 `_rebuild_shim_from_storage`、`_hf_build_shim`/`_ms_build_shim`/`_ollama_build_shim`、`_ollama_models_root`、`_quick_hash`;SP1 `EnvDetector`、`op_env_apply`。
全部产线代码进单文件 `aim.py`(新 `# ── Backup / Restore (SP3) ──` 段)。

---

## 2. `aim backup`

### 命令
`aim backup <dir> [--verify] [--json]` — 默认整库;`<dir>` = 目标目录(外置盘/另一路径)。

### 备份内容
- `<dir>/store/` ← 复制整个 `root.store_path`(所有 ingested + `aim download` 的扁平模型 = 全部真实字节)。
- `<dir>/aim-backup.json` ← 自包含清单:
```jsonc
{
  "aim_backup_version": 1,
  "created_at": "<ISO8601>",
  "source_root": "/Users/hjma/AI",
  "models": [ <每个 registry 模型的 to_dict(),含 storage 标注> ],
  "sources": { <config.get("sources", {}) 的 managed_env 等> },
  "env": { <config.get("env", {})> },
  "store_files": [ {"path": "store/<rel>", "size": int, "quick_hash": str} ]
}
```
- **排除(可再生,不入备份)**:加载壳(HF `hub/models--*`、Ollama `blobs/manifests`、MS 缓存目录)、HF `xet/`/logs、各工具缓存。它们均能在还原时从 `storage` 标注重建。

### 幂等同步（`_sync_store_dir`）
逐文件复制 `store/` → `<dir>/store/`;目标已存在且**同 `st_size`** 则跳过(`--verify` 时改比 `quick_hash`)。返回 (copied, skipped)。模型权重不可变 → 二次备份近乎瞬时。

### 未摇入原生模型告警
备份前扫描 registry:存在 `native_cas=True` 的模型(尚未 `aim ingest`)→ 它们不在 store、不会被备份 → 打印告警:
"N 个原生模型未摇入,运行 `aim ingest --all-native` 后再备份可纳入"(不阻断:其字节在工具缓存里、可重新下载)。

---

## 3. `aim restore`

### 命令
`aim restore <dir> [--root <id>] [--apply-env] [--verify] [--json]`

### 流程
1. `_read_backup_manifest(<dir>)`(校验 `aim_backup_version`)。
2. 定目标 root(`--root` 或 `get_primary_root`);target store = `root.store_path`。
3. **重建 store/**:`_sync_store_dir(<dir>/store, target_store, verify)`。
4. **导入 registry**:每个 manifest.models → `ModelEntry.from_dict`,`canonical["root"]=目标 root.id`(`canonical["path"]` 是 store 相对路径,可移植直接用),`registry.add`;末尾 `registry.save()`。
5. **重算 + 重建壳**:对每个有 `storage.shims` 的模型:
   - `_retarget_shim_locations(entry, EnvDetector())` 用目标机缓存根重算每个 shim 的 `location`:
     - `hf-cas` → `EnvDetector.cache_dir("huggingface")`(=`$HF_HOME/hub`)`/ f"models--{org}--{repo}"`(`org/repo`=`reconstruct.repo_id`)
     - `ollama-cas` → `EnvDetector.cache_dir("ollama")`(=`$OLLAMA_MODELS`)`/ "manifests" / reconstruct.manifest_rel`
     - `ms-dir` → `EnvDetector.cache_dir("modelscope")`(=`$MODELSCOPE_CACHE`)`/ "models" / org / reconstruct.dir_name`
   - 然后 `_rebuild_shim_from_storage(config, entry)`(SP2)从 store 建壳到新 `location`。
   - 单模型壳重建失败 → 收集错误、继续其余、末尾汇总(不中断)。
6. **恢复源配置**:把 manifest.sources 的 `managed_env` 并入 `config["sources"]`(目标机获得同样的 HF_ENDPOINT/加速等)。
7. **env**:`EnvDetector` 检测目标机并**打印**建议值(含恢复的 managed_env);`--apply-env` 才调 `op_env_apply` 写入,否则提示"运行 `aim env apply` 应用"。
8. **verify**:对重建的壳做一遍校验(复用 SP2 verify 的 shim 检查),汇总 ok/失败。

### 安全
- 幂等可重跑;store 跳过同大小、**不删除**目标多余文件(只新增/覆盖壳)。
- store 必须先于壳重建完成(壳指回 store);壳重建是从 store 派生,失败不丢数据(store 已在)。

---

## 4. 测试策略（stdlib unittest;全合成,无需真实工具）
- **`_sync_store_dir`**:复制 + 幂等(重跑跳过同大小)+ `--verify` 检出大小相同但内容不同(quick_hash 不同)→ 重新复制。
- **`op_backup`**:合成 store+registry(含 HF/MS/ollama `storage`)→ 断言 `<dir>/store/` 镜像、manifest 含全部 models+sources+env+完整性表;幂等(二次跳过);存在 `native_cas=True` 模型 → 打印告警。
- **`_retarget_shim_locations`**:注入 `EnvDetector`(给定目标缓存根)→ 断言 HF/MS/ollama 三种 `location` 重算正确(指向目标缓存根,而非源路径)。
- **restore 往返(重点)**:源机:合成 store + ingested 模型(HF/MS/ollama 壳 + registry)→ `op_backup` 到 `<dir>` → 模拟**异机**(新 target root + **不同**缓存位置,经 `EnvDetector` 注入 shell 值)→ `op_restore` → 断言:target store 重建、registry 导入(canonical.root=目标)、壳建在**目标缓存位置**且解析到 target store、env 被检测打印(未写 shell)。
- **restore 幂等**:二次 restore 不报错、结果稳定。
- **`--apply-env`**:临时 HOME 下断言 `~/.aim/env.sh` 写入 + rc 接线(复用 SP1)。
- **单壳失败不中断**:打桩使某模型壳重建抛错 → 其余模型仍还原、末尾报告该错误。
- e2e:真实工具在则可跑(否则 skip)。

---

## 5. 验收标准
1. `aim backup <dir>` 把 store/ 镜像到 `<dir>/store/` + 写自包含 `aim-backup.json`(含全部 registry 模型与 storage 标注);二次运行幂等(跳过同大小)。
2. 备份排除可再生壳/缓存(备份体积 ≈ store 真实字节 + 小 JSON)。
3. 存在未摇入原生模型时备份给出明确告警。
4. `aim restore <dir> --root <异机root>` 在**不同缓存路径**下:重建 store、导入 registry、把 HF/Ollama/MS 壳重建到**目标机**对应缓存位置且解析到 target store。
5. restore 默认**不修改 shell 配置**;`--apply-env` 才写;两种情况都打印建议 env。
6. restore 幂等;单模型壳失败不中断整库还原并在末尾报告。
7. 全部单测通过(`make test`),无第三方依赖。
