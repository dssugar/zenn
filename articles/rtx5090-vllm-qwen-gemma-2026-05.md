---
title: "RTX 5090 (32GB) 単機で 30B 級 LLM 4 モデルを vLLM 比較するまで ― Qwen 3.6 / Gemma 4"
emoji: "🚀"
type: "tech"
topics: ["vllm", "rtx5090", "qwen", "gemma", "blackwell"]
published: false
---

## はじめに

自宅の RTX 5090 (32GB) 1 枚で、30B 規模の LLM をそれなりの速度で動かしたい。
2026-04 に Qwen 3.6 系 (27B dense + 35B-A3B MoE) がリリースされたのを機に、
これまで動かしていた **Gemma 4 31B Turbo NVFP4** から別モデルに切り替えるかを検討するため、
**Qwen 3.6 / Gemma 4 の dense / MoE 計 4 モデル**を vLLM 0.20.1 で実測しました。

「Blackwell consumer GPU (sm120) で MoE と hybrid SSM モデルを動かすと何が起きるか」
を中心に、各モデルの起動時のハマりどころと回避策を残します。
特に **Qwen 3.6-35B-A3B (MoE) で踏んだ 3 連続の落とし穴**は、同条件 (RTX 5090 + vLLM 0.20.1) で
試す方の道標になるはずです。

:::message
本記事は **Phase 1 (起動 triage) と Phase 2 (チューニング) まで完了した時点の中間記録**です。
Phase 3 (concurrency / spec dec acceptance / 日本語品質 bench) は別記事に分ける予定。
速度数値はすべて短文 (~30 token) prompt × 256 token 出力なので、長文 ctx で逆転がある可能性に注意。
:::

## TL;DR

| モデル | quant | 最良構成 (本記事時点) | tok/s | 起動可否 |
|---|---|---|---|---|
| Gemma 4 31B Turbo (dense) | NVFP4 | EAGLE-3 spec dec | 54.7 | ✓ |
| Gemma 4 26B-A4B (MoE) | NVFP4 | baseline (spec dec で逆効果) | **162.8** | ✓ |
| Qwen 3.6-27B (dense, hybrid SSM) | NVFP4 | baseline (MTP で逆効果) | 52.6 | ✓ |
| Qwen 3.6-35B-A3B (MoE) | NVFP4 | **MTP k=2** (3 つの workaround 後) | **240.1** | ✓ (要 workaround) |

- MoE 2 種は active params が 3〜4B で **dense 30B 帯の 3 倍速い**
- Qwen 3.6-35B-A3B MoE は素直に動かないが、3 つのフラグで動かせる
- **短文では spec dec はほとんど効かない** — EAGLE-3 / MTP / ngram のいずれも +2% かマイナス。例外は Qwen MoE 内蔵 MTP (+25%)

## 検証マシン

| Part | Spec |
|---|---|
| CPU | AMD Ryzen 9 7900 (12C/24T) |
| RAM | 64 GB DDR5-6000 |
| GPU | NVIDIA RTX 5090 (32 GB VRAM, ZOTAC) |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-15-generic |
| Python | 3.14.4 (system) → uv で 3.12.13 venv |
| CUDA | Toolkit 13.1 + 12.9 併設 (nvcc は 13.1) |
| Driver | nvidia-driver-595-open (DKMS, 595.58.03) |
| vLLM | 0.20.1 (リリース 2026-05-04) |

:::message alert
Ubuntu 26.04 LTS の Python 3.14 は AI ecosystem には早すぎ気味です。
deadsnakes PPA も 26.04 対応待ち (2026-05 時点)。venv は `uv python install 3.12` で別系統に立てるのが堅いです。
:::

PCIe slot は **Gen5 x16** を BIOS で明示。デフォルト Gen4 のままだとここで頭打ちします
(`lspci -vv -s 00:01.1 | grep LnkCap` が `32GT/s` なら OK、`16GT/s` なら BIOS が reset されている可能性)。

## 候補 4 モデル

選定軸は **(Gemma vs Qwen) × (Dense vs MoE) の 2x2**。各モデルとも NVFP4 quant が公開されているのも条件:

| 略称 | repo |
|---|---|
| Gemma31 (dense) | `LilaRest/gemma-4-31B-it-NVFP4-turbo` |
| GemmaMoE | `nvidia/Gemma-4-26B-A4B-NVFP4` (公式) |
| Qwen27 (dense) | `unsloth/Qwen3.6-27B-NVFP4` |
| QwenMoE | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` |

Qwen 3.6 系は両方とも:
- **hybrid attention**: 数層おきに full_attention (GQA) を挟み、残りは linear_attention (Gated DeltaNet/SSM)
- **262k native context**
- **MTP (Multi-Token Prediction) 内蔵**: drafter weights が `model_mtp.safetensors` に同梱、別途用意不要

Gemma 4 はマルチモーダルですが本記事では **`--limit-mm-per-prompt`** で text-only 起動して比較します。

公的 benchmark (参考):
- Qwen 3.6-27B: MMLU-Pro 85.2 / GPQA Diamond 87.8 / **Terminal-Bench 2.0 = 59.3 (Claude 4.5 Opus と同点)** ([Qwen blog](https://qwen.ai/blog?id=qwen3.6-27b))
- Qwen 3.6-35B-A3B: MMLU-Pro 85.2 / GPQA Diamond 86 / Terminal-Bench 2.0 = 51.5

agentic coding 用途なら **27B (dense) のほうが MoE より公的スコア上**という意外な構図があります。

## 共通の bench harness

3 reps avg、stream で TTFT 計測:

```python
prompt = "RTX 5090 (32GB) で 27B 規模の LLM を NVFP4 で動かす利点を、3 つ簡潔に教えてください。"
# system: "あなたは丁寧な日本語アシスタントです。"
# temperature=0, max_tokens=256, stream=True
# 同一 prompt × 同一 system message を 4 モデル全てに投入し、横比較。
```

prompt token 数は ~30、出力は 256 token 固定。短文なので spec dec の acceptance rate は低めに出ます。

## 各モデルの起動と最初のスコア

### Gemma 4 31B Turbo (現状の常用モデル、baseline 取り直し)

```bash
vllm serve LilaRest/gemma-4-31B-it-NVFP4-turbo \
    --quantization modelopt --max-model-len 16384 \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 4096 --enable-prefix-caching
```

結果: **TTFT 24.6ms / 53.5 tok/s** (compile + cudagraph 有効、init 56s)。

`head_dim=512` の full_attn 層が 10 個あるため、本機 sm120 では FA backend が選べず TRITON_ATTN
にフォールバックする (FA4 sm120 解禁待ち、[Issue #36865](https://github.com/vllm-project/vllm/issues/36865))。
それでも実用速度。

### Gemma 4 26B-A4B (MoE)

```bash
vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
    --max-model-len 16384 --max-num-seqs 32 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
```

結果: **TTFT 18.1ms / 162.8 tok/s** — Gemma31 dense の 3 倍。
active 4B/26B = 約 1/6.5 の実効計算量がそのまま decode 速度に乗る MoE の典型挙動。

config inspection で気付いた点:
- `text.use_bidirectional_attention: "vision"` → mm 入力 0 にした text-only 経路で gating 回避
- `quantization_config` に `kv_cache_scheme {num_bits: 8, type: float}` 明記、fp8 KV 強制
- 30 層: 25 sliding (window 1024) + 5 full、head_dim 256

### Qwen 3.6-27B (dense, hybrid SSM)

config を読むと、本機 sm120 にとって **3 つの障壁を全部回避**する作りでした:

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "text_config": {
    "num_hidden_layers": 64,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", ...]
  },
  "quantization_config": {
    "format": "nvfp4-pack-quantized",
    "kv_cache_scheme": null
  }
}
```

- `head_dim=256` (Gemma 31B の 512 と違う、FA backend が選べる)
- `kv_cache_scheme: null` = bf16 KV (fp8 KV gate に詰まらない)
- 64 層中 16 が full_attention (GQA)、48 が linear_attention (Gated DeltaNet/SSM) → KV cache が小さく long ctx に強い

vLLM 0.20.1 はこの hybrid SSM を `qwen3_5.py` / `qwen3_next.py` で実装済 (Qwen3-Next 80B-A3B 系の sibling)。

ただし起動して smoke すると 2 か所引っかかります。

#### enable_thinking が default ON

最初の応答で **思考プロセスが英語で延々**:

```
Here's a thinking process:
1. Analyze User Input:
- Hardware: RTX 5090 (32GB VRAM)
...
```

256 token 制限で truncate されて本文の日本語が出ません。
`chat_template.jinja` を読むと:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}
{%- endif %}
```

vLLM の `chat_template_kwargs` で disable:

```json
{
  "model": "unsloth/Qwen3.6-27B-NVFP4",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

#### `--enforce-eager` だと 19.8 tok/s しか出ない

最初に保守的に `--enforce-eager` で起動したら **19.8 tok/s**。eager 外す
(= compile + cudagraph 復活) と:

- compile time: 34.08s
- CUDAGraph capture: PIECEWISE 11 + FULL 7 = 18 graphs
- decode: **52.5 tok/s = 2.65x speedup**

SSM/Mamba 系は per-layer の small op が多いため、cudagraph による host-side overhead 削減が劇的に効きます。
**hybrid SSM モデルでは eager mode は避ける**。

最終結果: **TTFT 67ms / 52.6 tok/s** (Gemma 31B dense と互角)。

### Qwen 3.6-35B-A3B (MoE) ― 落とし穴を 3 つ越えるまで

ここが本記事の山場。Qwen MoE は素直に起動できず、3 つの問題を順番に解いていきました。

#### 落とし穴 1: NVCC が system RAM を食い尽くして OOM-killer 発動

```bash
vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
    --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
```

**8 分後に engine init failed**。GPU OOM の表示なし、tmux session も死亡。

`dmesg | grep -i oom` で実体判明:

```
Out of memory: Killed process 212895 (cicc) total-vm:5199204kB, anon-rss:4539320kB
Out of memory: Killed process 212891 (cicc) total-vm:6126948kB
```

**`cicc` は NVCC の internal compile pass**。FlashInfer の `fused_moe_120` (sm120 専用) が
最大 80 個の cuda kernel を JIT compile する際、NVCC が **並列で 6 個立ち上がり、各 ~5GB anon memory を要求**。
system 64GB RAM が足りずに OOM-killer が cicc を殺す → kernel build 失敗 → engine init 落ちる。

ログには cuda kernel コンパイラの長い行 (`[3/80] /usr/local/cuda/bin/nvcc ...`) が並ぶだけで、
それが原因とは見えません。`dmesg` を見るのが確実です。

回避:

```bash
export MAX_JOBS=1
export NVCC_THREADS=1
```

sequential build にすると初回起動は 8〜10 分かかりますが、`~/.cache/flashinfer/0.6.8.post1/120f/`
に kernel cache されるので 2 回目以降は瞬時。

:::message
RTX 5090 (sm120) + FlashInfer の組み合わせを使う model 全般で発生し得ます。
nvcc を多並列で走らせる setuptools / cmake build 一般にも応用できる workaround。
:::

#### 落とし穴 2: torch.compile AOT trace で `NoneType has no attribute 'size'`

NVCC RAM を解決して再起動 → 別エラー:

```
File "vllm/model_executor/models/qwen3_5.py", line 695, in forward
    hidden_states = self.language_model.model(
File "vllm/compilation/decorators.py", line 533, in __call__
    output = self.aot_compiled_fn(self, *args, **kwargs)
File "torch/_dynamo/aot_compile.py", line 224, in __call__
    return self.fn(*args, **kwargs)
File "vllm/model_executor/models/qwen3_next.py", line 495, in forward
    def forward(
File "torch/_dynamo/utils.py", line 5033, in call_size
    return x.size(i)
AttributeError: 'NoneType' object has no attribute 'size'
```

`@support_torch_compile` (`Qwen3NextModel.forward` に付いてる) の **AOT trace** が、
`input_ids=None` かつ `inputs_embeds=None` のシンボリック placeholder で
`x.size()` を呼ぼうとして落ちています。

不思議なことに dense (Qwen 27B) では同じ forward が問題なく動きます。MoE 固有の
何か (decoder layer 違い、aux_hidden_state 周り) が trace 中に None placeholder を経由している模様。

回避: **AOT trace のみ無効化**:

```bash
--compilation-config '{"ir_enable_torch_wrap":false}'
```

これで `mode=VLLM_COMPILE` (3) と CUDAGraph capture (PIECEWISE 11 + FULL 7 = 18 graph) は維持しつつ、
torch.compile の AOT trace 経路だけ skip します。

結果: **23.6 → 190.8 tok/s = 8.1x speedup**。Gemma MoE (162.8) も超えて全モデル中 best へ。

:::message alert
これは vLLM 0.20.1 の Qwen3_5MoeForConditionalGeneration 固有のバグの可能性が高く、
上流に PR を出すべき issue です。本記事執筆時点では未報告のはずなので、再現する方は
issue 立てや patch 提案の余地があります (本記事の落とし穴 2 を参考にどうぞ)。
:::

#### 落とし穴 3: MTP 追加で KV pool が逼迫

ここで MTP 内蔵 spec dec を試行:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

MoE では問題なく起動して **240.1 tok/s** (no-MTP 比 +25.8%)。

ところが Qwen 27B (dense) で同じフラグを足すと CUDA OOM:

```
torch.OutOfMemoryError: CUDA out of memory.
ValueError: To serve at least one request with the models's max seq len (16384),
(1.71 GiB KV cache is needed, which is larger than the available KV cache memory (1.23 GiB).
```

dense は full_attn 層が 16 (MoE は 10) で **KV cache が ~60% 重い** + MTP drafter weights 追加で、
util 0.92 では収まりません。

```bash
# Qwen 27B + MTP の最終 config
--max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.95
```

これで起動。だが結果は **39.7 tok/s** = MTP off (52.6) より **24% 遅い**。
acceptance rate が低いまま spec dec の overhead だけ食う、よくある負けパターン。

→ **Qwen MoE の MTP 内蔵は強い (+25%) が、Qwen 27B Dense の MTP 内蔵は弱い**。
仮説:

- unsloth の NVFP4 変換時に MTP weights の精度が落ちている
- 短文 prompt では drafter が context 不足

長文 prompt + concurrency 設定で再度挙動が変わるかもしれません (Phase 3 で要検証)。

## Phase 1 + 2 の比較表 (短文 prompt)

| モデル | Phase 1 baseline | Phase 2 best | 改善 |
|---|---|---|---|
| Gemma 31B (dense) | 53.5 (no-spec) | 54.7 (EAGLE-3 k=3) | +2% |
| Gemma MoE | **162.8** (no-spec) | **162.8** (ngram k=4 = 121, -26%) | 0% |
| Qwen 27B (dense, hybrid) | 52.6 (no-spec) | **52.6** (MTP k=2 = 39.7, -24%) | 0% |
| Qwen MoE | 23.6 (eager) | **240.1** (no-AOT + MTP k=2) | **+917%** |

短文 prompt では spec dec の旨味が出にくい (EAGLE-3 / MTP / ngram どれも +2% かマイナス)。
**spec dec の真価は longer ctx (1k-4k 入力) で出る**ので、Phase 3 で `vllm bench serve` で検証予定です。

## 速度向上手段の整理 (本機環境、2026-05-05 時点)

### 即時可能 (config 変更のみ)

| 手段 | 効果 | 注意 |
|---|---|---|
| compile + cudagraph (default) | 2-3x decode | QwenMoE は AOT bug → `ir_enable_torch_wrap:false` |
| spec dec (EAGLE/MTP/N-gram) | +0〜+50% | 短文では効かないことも (本記事 Phase 1+2 参照) |
| KV cache fp8 (`--kv-cache-dtype fp8`) | KV 1/2 → ctx 2x | Gemma 系で適用済 |
| KV cache nvfp4 (`--kv-cache-dtype nvfp4`) | KV 1/4 → ctx 4x | [PR #40177](https://github.com/vllm-project/vllm/pull/40177) (2026-05-01 merged)、SM100 明記、SM120 で動くか未検証 |
| `--max-num-batched-tokens` 4096-8192 で sweep | medium decode +5-10% | mnbt 8192 は long-ctx OOM 注意 |

### 軽実装 (~1h)

| 手段 | 用途 |
|---|---|
| `MAX_JOBS=1 NVCC_THREADS=1` | NVCC RAM OOM 回避 |
| `--hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'` | Gemma 系 mm_prefix gating 解除試行 |
| `--limit-mm-per-prompt` で全 mm 入力 0 | text-only mode、vision encoder profiling skip |

### 待ち (上流次第)

| 項目 | リンク | 影響 |
|---|---|---|
| FA4 SM120 完全対応 | [Issue #36865](https://github.com/vllm-project/vllm/issues/36865) | Gemma 31B head_dim=512 の FA backend 解禁 |
| NVFP4 KV cache SM120 動作確認 | [PR #40177](https://github.com/vllm-project/vllm/pull/40177) | long ctx +20-40% |
| TRITON_ATTN non-causal 対応 | (上流 PR 待ち) | DFlash 起動可能化 |
| AOT compile None bug fix | 自前 patch 候補 (本記事 落とし穴 2 で workaround 済) | 上流 PR 価値あり |

## 中間結論

短文 prompt 限定の Phase 1+2 結果から言えること:

1. **MoE 2 種は dense の 3 倍速い** — active 4B/26B (Gemma) や 3B/35B (Qwen) 比でほぼ妥当
2. **Qwen MoE は素直には動かないが、3 フラグで動く**:
   - `MAX_JOBS=1 NVCC_THREADS=1` (build 時 RAM)
   - `--compilation-config '{"ir_enable_torch_wrap":false}'` (AOT bug)
   - `--limit-mm-per-prompt` で全 mm 0 (vision profiling skip)
3. **動かせれば Qwen MoE が現状最速 (240 tok/s) — Gemma MoE よりも 17% 速い**
4. **短文では spec dec はほとんど効かない** (例外: Qwen MoE 内蔵 MTP +25%)。
   長文 ctx では別の絵が出るはずで、Phase 3 で再検証

常用モデルの切り替え判断は Phase 3 (concurrency / 日本語品質 / long ctx 限界) を待ってからするのが妥当ですが、
**短文 prompt 時点では Qwen MoE が Gemma 31B Turbo + EAGLE-3 (本記事 54.7 tok/s、別シナリオの実測で 93 tok/s) を上回る**ことは確定。

## 次の記事 (予定)

- **Phase 3**: `vllm bench serve --dataset-name {random, spec_bench}` で longer ctx + concurrency 曲線を取り、spec dec acceptance rate を 4 モデル × 3 spec dec method で表化
- **日本語品質**: bench-jp.py を実装し、20-30 個の日本語 prompt 別 (敬語切替 / 翻訳ニュアンス / 慣用句 / 難読漢字 / 文化的文脈 / 方言 / 技術 docstring / 同音異義 / RP / 長文要約) を 4 モデルに投げて LLM-as-judge で side-by-side
- **long ctx 限界**: 各モデル `--max-model-len {32k, 64k, 131k, 262k}` で boot 限界を実測

## 参考リンク

- vLLM: <https://github.com/vllm-project/vllm>
- Qwen 3.6 公式 blog: <https://qwen.ai/blog?id=qwen3.6-27b>
- Qwen 3.6-27B (HF): <https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4>
- Qwen 3.6-35B-A3B (HF): <https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4>
- Gemma 4 26B-A4B NVFP4 (HF): <https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4>
- Gemma 4 31B Turbo NVFP4 (HF): <https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo>
- 本機の hw 構成と運用詳細: [dssugar/dai-gpu-server (PRIVATE)](https://github.com/dssugar/dai-gpu-server)
