---
title: "RTX 5090 (32GB) で 30B 級 LLM を vLLM 比較 (Phase 3+4): 各モデルの本機上限を引き出すチューニング記録"
emoji: "📈"
type: "tech"
topics: ["vllm", "rtx5090", "qwen", "gemma", "speculative-decoding"]
published: false
---

## はじめに

[前回の記事](https://zenn.dev/dssugar/articles/rtx5090-vllm-qwen-gemma-2026-05) で
RTX 5090 + vLLM 0.20.1 上に **Gemma 4 / Qwen 3.6 の dense + MoE 計 4 モデル**を起動し、
短文 (~30 token) prompt × 256 token 出力で速度比較しました。
結論は **Qwen 3.6-35B-A3B (MoE) が 240 tok/s で勝者**、ただし
「短文 prompt では spec dec の真価が出にくい」「context が長いと順位が変わる可能性」
という宿題が残っていました。

本記事の目的は **「常用構成を選ぶ」ことではなく、各モデルが本機 (RTX 5090 32GB) で
出せる上限値と、それを引き出すチューニング技法を per-model に詰める** ことです。
比較表ではなく **モデル × 上限値 × その knob** をワンセットで残します。

具体的には **`vllm bench serve` を中心に 5 軸の負荷を掛けて per-model 上限を取り**ます:

1. **longer context** — `random` dataset で input 1k / 4k / 8k / 16k / 32k / 40k
2. **concurrency** — c=1 / 4 / 8 / 16 / 32 で aggregate throughput
3. **natural language acceptance** — `spec_bench` dataset (coding / qa / summarization)
4. **MTP num_speculative_tokens の k sweep** — k=2 〜 k=5
5. **max-model-len 上限実測** — 4 モデル全てで本機 32GB に乗る ctx 上限

結果として、**短文 chat では見えなかった発見と、各モデルの「本機上限」プロファイル**が出てきました。

## TL;DR — 4 モデルの本機 RTX 5090 (32GB) 上限プロファイル

| Model | 本機 decode 上限 | 本機 mml 上限 | 本機 concurrency 上限 |
|---|---|---|---|
| Gemma 4 31B Turbo + EAGLE-3 **k=4** | qa **100.4 tok/s** | **58,592** | (未測) |
| Gemma 4 26B-A4B (MoE, no spec) | c=1 149 / spec_bench qa 154 | **262,144** = native | c=32 で **2,318 tok/s aggregate (15.5x)** |
| Qwen 3.6-27B (dense, hybrid SSM) | 1k 50 / 40k **23.7** (TPOT +8% only over 40x ctx) | **41,552** | (低 prio) |
| Qwen 3.6-35B-A3B + MTP **k=4** | spec_bench qa **272 tok/s** | **126,208** | c=4 で 336 (1.91x scale) |

ハイライト:

- **MTP は input semantic 非依存に効く** (random tokens でも acc 47%、natural language で **k=4 acc len 3.28**)
- **Qwen MoE MTP の本機ピークは k=4** (k=2: 240 / k=3: 260 / **k=4: 272** / k=5: 261 で peak-out)
- **Gemma MoE concurrency は 32 まで scale** (per-stream 79-90 tok/s 維持で aggregate 2,318)
- **Gemma MoE は本機で full 262k context が乗る** (sliding window 1024 が KV を ctx 非依存にする)
- **Qwen 27B SSM benefit は long ctx でしか visible** (TPOT 1k → 40k で +8% only)
- **Gemma 31B + EAGLE-3 の真の k 上限は k=4** (現常用 default の k=3 から +8%)

## TL;DR (workload 別の速度順位)

| Workload | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| 短文 chat (Phase 1+2) | Qwen MoE 240 | Gemma MoE 163 | Gemma 31B EAGLE-3 55 | Qwen 27B 53 |
| `random` 1k single | **Qwen MoE+MTP 176** | Gemma MoE 149 | Gemma 31B EAGLE-3 53 | Qwen 27B 50 |
| `random` 4k single | **Qwen MoE+MTP 149** | Gemma MoE 138 | Gemma 31B EAGLE-3 48 | Qwen 27B 47 |
| `random` 1k c=4 aggr | **Gemma MoE 446** | Qwen MoE+MTP 337 | Qwen 27B 170 | Gemma 31B 151 |
| `spec_bench` coding | **Qwen MoE+k4 251** | Gemma MoE 154 | Gemma 31B EAGLE-3 **k=4 127** | Qwen 27B 51 |
| `spec_bench` qa | **Qwen MoE+k4 272** | Gemma MoE 154 | Gemma 31B EAGLE-3 **k=4 100** | Qwen 27B 52 |
| `spec_bench` summ | **Qwen MoE+k4 240** | Gemma MoE 149 | Gemma 31B EAGLE-3 **k=4 87** | Qwen 27B 51 |
| `random` 1k **c=32 aggr** | **Gemma MoE 2,318** | (他モデル未測) | | |

- **single-stream は Qwen MoE + MTP k=4 が全 workload 勝者** (qa 272 tok/s = 短文 240 から +13%)
- **concurrency 4 は Gemma MoE が勝者** (442 tok/s aggregate)
- **Qwen MoE は c=4 で 1.91x しかスケールしない** vs Gemma MoE 2.97x (MTP の batched verify cost が支配的)
- **MTP は input semantic 非依存に効く** (random tokens でも acc 47-54%、natural language で 72-77%)
- **EAGLE-3 (learned drafter) は random で acc 11% に落ちる** = MTP (intrinsic) と挙動真逆

## 復習: 4 モデル + 各々の Phase 2 best 構成

| 略称 | repo | Phase 2 best 構成 |
|---|---|---|
| Gemma 31B EAGLE-3 | `LilaRest/gemma-4-31B-it-NVFP4-turbo` | EAGLE-3 spec dec k=3 (`serve-default.sh`) |
| Gemma MoE | `nvidia/Gemma-4-26B-A4B-NVFP4` | no spec dec (ngram k=4 で逆効果だった) |
| Qwen 27B | `unsloth/Qwen3.6-27B-NVFP4` | no spec dec (MTP k=2 で -24% 逆効果だった) |
| Qwen MoE | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | no-AOT compile + MTP k=2 (本記事で k=4 まで sweep) |

`vllm bench serve` の使いこなし、SpecBench データセット入手、ローカル patch 等は次節以降。

## 共通の bench harness

```bash
# random dataset (純 decode throughput)
vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions --model <MODEL> \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 256 \
    --num-prompts 10 --max-concurrency 1 \
    --seed 42 --save-result --result-dir bench-phase3 \
    --result-filename <label>-rand1k-c1.json

# spec_bench dataset (natural language acceptance)
vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions --model <MODEL> \
    --dataset-name spec_bench \
    --dataset-path /path/to/question.jsonl \
    --spec-bench-category coding \
    --spec-bench-output-len 256 \
    --num-prompts 10 --max-concurrency 1 \
    --seed 42
```

各モデルで `random {1k-c1, 4k-c1, 1k-c4}` の 3 run、`spec_bench {coding, qa, summarization}` の 3 run、計 6 run。
モデル切替には `tmux kill-session -t vllm` → 該当 serve script を tmux で再起動。

### 落とし穴 1: vLLM 0.20.1 SpecBench クラスの self 抜けバグ

`spec_bench` を最初に走らせるとこれで落ちます:

```
TypeError: SpecBench.sample() takes 0 positional arguments but 1 was given
```

`vllm/benchmarks/datasets/datasets.py` の `SpecBench.sample` の定義を見ると:

```python
def sample(
    **kwargs,
) -> list[SampleRequest]:
    return super().sample(**kwargs)
```

**`self` が抜けてます**。Python が `self` を positional として渡すので "0 positional accepted" エラー。
ローカル patch:

```python
def sample(
    self,
    **kwargs,
) -> list[SampleRequest]:
    return super().sample(**kwargs)
```

これは vLLM 上流の bug の可能性が高いです (本記事執筆時点で issue/PR を検索しましたが該当なし)。

### 落とし穴 2: `spec_bench` の dataset path と category 名

`--help` には spec_bench dataset の docstring に `mt_bench` / `humaneval` という category 名が出てきますが、
**これは Spec-Bench 公式 (hemingkx/Spec-Bench) の本物の category 名と一致しません**。
本物は `coding / qa / math_reasoning / summarization / translation / rag / writing / ...` の 13 カテゴリ。

dataset 自体も `--dataset-path` で明示する必要があり、入手は:

```bash
mkdir -p datasets && cd datasets
wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
```

(480 prompts、上記 13 categories)

### 落とし穴 3: pandas が venv に入ってない

`spec_bench` は jsonl を `pd.read_json` で読みます。`vllm[bench]` extra に含まれていなかったため別途 install:

```bash
uv pip install --python /home/dai/venvs/vllm-pr38891-base/bin/python pandas
```

## 結果 1: random dataset (純 decode throughput)

10 prompts × 単発、output 256 token:

| Model | rand1k c1 (tok/s) | rand4k c1 (tok/s) | rand1k c4 aggr (tok/s) | spec acc / len |
|---|---|---|---|---|
| Gemma 31B EAGLE-3 | 52.5 | 48.3 | 151.3 | 11.6% / 1.35 |
| Gemma MoE (no spec) | 149.2 | 138.2 | **442.7** | n/a |
| Qwen 27B | 50.2 | 47.4 | 170.2 | n/a |
| Qwen MoE + MTP k=2 | **176.1** | **149.2** | 336.5 | **47-54% / 1.95-2.07** |

注目すべきは spec dec acceptance rate です:

- **Gemma 31B EAGLE-3**: random で acc **11.6%** (短文 chat で 25-40% 出ていたのと比べて大幅低下)
- **Qwen MoE MTP k=2**: random で acc **47-54%** (短文 chat 比でほぼ同程度)

これは **MTP と EAGLE-3 の構造的差**です。EAGLE-3 は別 trained drafter (本機では 2.235B BF16) で、
学習時の分布を頼りに input から次 token を予測します。input が乱数だと当然予測精度が落ちます。

一方 MTP は **model 自身の hidden state から次 token を予測する intrinsic head** です。
input が何であろうと「自分が今どんな token を出力するか」は自分の hidden state を見れば分かるので、
random でも効きます (acc 47%)。

これは 1 つ目の発見:

> **MTP は input semantic 非依存に効く。EAGLE-3 (learned drafter) は input semantic に依存する。**

## 結果 2: spec_bench dataset (natural language acceptance)

3 categories × 10 prompts、output 256 token:

| Model | coding tok/s | qa tok/s | summ tok/s | qa acc % | qa acc len |
|---|---|---|---|---|---|
| Gemma 31B EAGLE-3 | 126.0¹ | 92.8 | 80.2 | **39.78** | 2.19 |
| Gemma MoE (no spec) | 154.1 | 154.2 | 149.4 | n/a | n/a |
| Qwen 27B (no spec) | 51.3 | 51.6 | 50.9 | n/a | n/a |
| Qwen MoE + MTP k=2 | **246.0** | **239.6** | **222.5** | **72.70** | 2.45 |

¹ 1 request fail (server warmup タイミング、acc 数値は無出力)

natural language では:

- **Gemma 31B EAGLE-3**: random の 11% から **39.78%** に大幅向上 (RedHat 公開数値とほぼ一致)
- **Qwen MoE MTP k=2**: random の 47% から **72.70%** に向上 (RedHat 表を上回る)

そして MTP の異常さは acceptance length に出ます。Qwen MoE k=2 で acc length **2.45** =
2 つの spec を **平均で 1.45 個 hit している** = ほぼ毎 step で 2.45 token decode。

per-position を見ると:
- Position 0 (1 個目の spec): **82%** hit
- Position 1 (2 個目の spec): **64%** hit

**両 spec を hit する確率が 60-70%** = MTP は単に length 2 の効果を取るだけでなく、
**毎 step が事実上 3 token decode に近い**状態。

これが 2 つ目の発見:

> **Qwen MoE の MTP intrinsic head は、natural language で acceptance 70%+ を出せる強力な spec dec。**

そして 3 つ目の発見:

> **現状の常用 (Gemma 31B + EAGLE-3) は Gemma MoE 単体に負ける。**

`spec_bench qa` で見ると:
- Gemma MoE (no spec): **154 tok/s**
- Gemma 31B + EAGLE-3 (acc 40% / len 2.2): **93 tok/s**

EAGLE-3 が 40% も hit しているのに、active parameter が少ない MoE には敵わないのです。
spec dec は **MoE の active param 効率の前には霞む**。

## 結果 3: concurrency スケーリング (c=1 → c=4)

random 1024 で c=4、20 prompts:

| Model | c=1 (tok/s) | c=4 aggr | per-stream c=4 | scaling |
|---|---|---|---|---|
| Gemma 31B EAGLE-3 | 52.5 | 151.3 | 37.8 | 2.88x aggr / 0.72 per-stream |
| Gemma MoE | 149.2 | **442.7** | 110.7 | **2.97x aggr** / 0.74 per-stream |
| Qwen 27B | 50.2 | 170.2 | 42.6 | 3.39x aggr / 0.85 per-stream |
| Qwen MoE + MTP | 176.1 | 336.5 | 84.1 | **1.91x aggr** / 0.48 per-stream |

これが 4 つ目の発見:

> **single-stream で勝っていた Qwen MoE は concurrency 4 で順位が逆転、Gemma MoE が aggregate 442 tok/s でトップに。**

Qwen MoE の 1.91x scaling は明らかに低いです。仮説:

- **MTP の batched verify cost** が batch size に対して super-linear に増える (drafter forward は batch 跨ぎで分散しにくい)
- compile + cudagraph capture が batch dimension で展開されている可能性 (specific to no-AOT 経路?)

詳細は本記事の scope 外ですが、**production の負荷シナリオが c=2-4 で常時動く想定なら Gemma MoE が有利**、
**single-stream の chat 用途なら Qwen MoE + MTP が有利**、と用途で使い分ける構図が見えてきます。

## 結果 4: Qwen MoE MTP k sweep — k=5 まで sweep して peak 確定

`spec_bench` 3 categories × Qwen MoE で `num_speculative_tokens = {2, 3, 4, 5}`:

| k | coding tok/s | qa tok/s | summ tok/s | coding acc len | per-position coding |
|---|---|---|---|---|---|
| 2 | 246.0 | 239.6 | 222.5 | 2.55 | P0 85 / P1 70 |
| 3 | 245.8 | 259.7 | 235.0 | 3.08 | P0 82 / P1 69 / P2 58 |
| **4** | **251.0** | **272.0** | **239.5** | 3.49 | P0 82 / P1 68 / P2 54 / P3 45 |
| 5 | 249.1 | 261.3 | 233.2 | **3.87** | P0 84 / P1 67 / P2 54 / P3 46 / **P4 37** |

**k=4 が全カテゴリで勝者、k=5 で peak-out** です。qa で **272 tok/s** (Phase 1+2 の short-prompt 240 から +13%)。
acc length は k=5 で 3.87 まで伸びる (P4 hit 37% も取れる) が、**verify overhead が gain を上回る**。

これが 5 つ目の発見:

> **Qwen MoE の MTP は k=4 で peak、k=5 で減速。Phase 1+2 の k=2 default は最適ではなかった。**

(他モデルでも MTP / EAGLE-3 の k 増加は逆効果になることが多いです。詳細は次のセクション。)

なお k=5 で起動するときは `--max-num-seqs 32 → 16` への絞りが必要です:
draft slot が増えて activation memory が逼迫し、CUDAGraph capture 中に
`torch.OutOfMemoryError: Tried to allocate 816.00 MiB` で engine init が落ちます。

## 結果 5: Gemma 31B Turbo + EAGLE-3 にも実は上限がある (新 best k=4)

私の常用構成 (`serve-default.sh`) は **EAGLE-3 k=3 + mml=16384 + mnbt=6144 + fp8 KV** です。
これは過去の handoff (medium prompt 1.95k, output 256) で「k=4 は medium で gain なし、long で OOM」
という結論で固定したもの。**しかし短文 spec_bench 文脈ではこの結論は成り立ちません**:

`max-model-len 8192 / max-num-batched-tokens 4096 / fp8 KV / `--max-num-seqs 32` で k sweep:

| k | coding tok/s | qa tok/s | summ tok/s | acc len (qa) |
|---|---|---|---|---|
| 3 (現常用 default) | 126.0 | 92.8 | 80.2 | 2.19 |
| **4** | **127.4** | **100.4** | **87.4** | 2.41 |
| 5 | 125.8 | 94.3 | 78.7 | 2.43 |

**短文 spec_bench では k=4 で qa +8.2% / summ +9.0%** = Gemma 31B + EAGLE-3 の本機 best が更新されます。
k=5 で再び減速 (Qwen MoE と同じパターン)。

つまり **「常用構成の k は workload で再評価する価値がある」**。
production には medium prompt + 高 mml の serve script (k=3 のまま) と、短文 spec_bench / chat 用 (k=4)
を分けて持つ運用が筋が良さそう。

## 結果 6: Qwen 27B (hybrid SSM) — 4k では SSM benefit 見えず、**40k で顕在**

Phase 3 で「4k 程度では SSM benefit 出ず」と書いた宿題を 8k 〜 40k で回収します。

`max-model-len 40960 / max-num-seqs 4 / util 0.95` で random output 256:

| input | tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---|---|
| 1,024 | 50.2 | 165.5 | 19.36 |
| 4,096 | 47.4 | 424.3 | 19.51 |
| 8,192 | 43.7 | 848.3 | 19.63 |
| 16,384 | 37.3 | 1,768.0 | 19.97 |
| 32,768 | 27.3 | 4,118.6 | 20.63 |
| 40,704 | 23.7 | 5,451.3 | 20.92 |

注目すべきは **TPOT** です。**1k → 40k で 19.36 → 20.92ms = +8% (40x ctx 増)**。
decode 自体は事実上 flat、tok/s 低下はすべて prefill (TTFT 5.4 秒) 寄与。

これが 6 つ目の発見:

> **Qwen 27B hybrid SSM の本懐は decode TPOT が ctx 非依存。Long-ctx チャットボット用途で本格的に効く。**

なお Qwen 27B で `max-model-len 65536` を試すと KV cache OOM、
vLLM が `estimated maximum model length is 41552` と教えてくれるので、本機実用上限は **41,552**。
`bf16 KV` + `16/64 full_attn 層` (残り 48 層は GDN/SSM で KV 持たない) という構造、
**dense Gemma 31B (fp8 KV、10 full_attn) より少ない**のが意外。full_attn 層数が支配的。

## 結果 7: Gemma MoE の concurrency 上限は **c=32 で 2,318 tok/s**

Phase 3 で c=4 = 442 tok/s (2.97x scale) を見ましたが、c=8/16/32 まで伸ばすとどうなるか:

| concurrency | aggregate tok/s | per-stream | TPOT (ms) | aggr scale |
|---|---|---|---|---|
| 1 | 149.4 | 149.4 | 6.52 | 1x |
| 4 | 446.3 | 111.6 | 8.68 | 2.99x |
| 8 | 635.6 | 79.4 | 9.35 | 4.25x |
| 16 | 1,444.2 | 90.3 | 10.40 | **9.66x** |
| 32 | **2,317.8** | 72.4 | 12.91 | **15.51x** |

**c=32 で aggregate 2,318 tok/s** (`--max-num-seqs 32` cap、これを上げないと c=64 以降は queueing で頭打ち)。
TPOT は c=1 → c=32 で 6.52 → 12.91ms = 2x のみ、Gemma MoE active 4B param と sliding window 1024 KV
の効率が並列処理で素直にスケール。

per-stream tok/s が **c=8 で minimum (79.4) の後 c=16 で回復 (90.3)** という non-monotonic な挙動も観察。
CUDA graph capture sizes (`[1,2,4,8,16,24,32,...]`) との関係、別 sweep の余地ありそうです。

これが 7 つ目の発見:

> **Gemma MoE は本機で並列 32 まで素直にスケール、aggregate 2,318 tok/s。short-stream 単体 149 tok/s から 15.5x。**

## 結果 8: per-model max-model-len 上限実測

最後の軸として、4 モデルの **本機 32GB で実際に乗る context 上限**を実測します。
方法はシンプル: `max-model-len` を意図的に高く設定して boot し、vLLM が吐く OOM error
`estimated maximum model length is X` を信頼する (`probe-mml.sh` 化、~7 分 / model)。

| Model | 本機実用 mml | native ctx | 削減要因 |
|---|---|---|---|
| **Gemma MoE** (no spec) | **262,144 (= native ceiling)** | 262,144 | sliding window 1024 が KV を ctx 非依存に |
| **Qwen MoE + MTP k=4** | **126,208** (≈123k) | 262,144 | bf16 KV + MTP drafter + 10/40 full_attn |
| **Gemma 31B + EAGLE-3 k=3** | **58,592** (≈57k) | 131,072 | fp8 KV だが drafter co-resident + head_dim=512 |
| **Qwen 27B** | **41,552** (≈40k) | 262,144 | bf16 KV + 16/64 full_attn (MoE の 10/40 より重い) |

順位が **Gemma MoE > Qwen MoE > Gemma 31B > Qwen 27B**、native ctx と
本機実用 mml の比は Gemma MoE 1.0 / Qwen MoE 0.48 / Gemma 31B 0.45 / Qwen 27B 0.16。
**Gemma MoE は本機で full 262k context が乗る数少ない 30B 級モデル**になります。

これが 8 つ目の発見:

> **Gemma MoE の sliding window (1024 cap × 25/30 層) は long-ctx メモリ効率の本命 knob。
> dense Gemma 31B の 4.5x mml が乗る。**

## まとめ: per-model 本機上限プロファイル (workload 別)

| 用途 | 推奨候補 | 数値根拠 (本記事) | 必要な knob |
|---|---|---|---|
| 短文 chat / 単発 | Qwen MoE + MTP **k=4** | qa **272 tok/s** | NVCC RAM 回避 / no-AOT compile / `enable_thinking: false` |
| 並列負荷 (c=4 〜 c=32) | Gemma MoE (no spec) | **2,318 tok/s aggr @ c=32** | `limit-mm-per-prompt 0` / `--max-num-seqs 32` |
| 長 ctx (~40k) | Qwen 27B | TPOT 20ms 維持 (1k→40k +8%) | mml 40960 / max-num-seqs 4 / util 0.95 |
| 長 ctx (~123k) | Qwen MoE + MTP k=4 | mml 126,208 | 上記 Qwen MoE knob 同 |
| 超長 ctx (~262k) | Gemma MoE | mml 262,144 (= native) | 上記 Gemma MoE knob 同 |
| 既存 EAGLE-3 構成を最大化 | Gemma 31B + EAGLE-3 **k=4** | qa 100.4 tok/s (現 default の +8%) | mml ≤8192 / mnbt 4096 / fp8 KV 既存 + k=4 |

:::message
**本記事は速度数値だけです**。日本語出力品質、unsloth NVFP4 quant の精度劣化、
実運用 workload 比率は別記事で扱います (本記事の数値だけで採用判断はしない方針)。
:::

## 上流 PR 候補 (本記事で発見した 2 件)

### 1. vLLM 0.20.1 SpecBench クラスの self バグ

`vllm/benchmarks/datasets/datasets.py:2367` の `def sample(**kwargs)` に `self` 引数追加。
patch は本記事の「落とし穴 1」参照。本記事の k=2/3/4/5 / 4 モデル全部のベンチを同 patch で完走 (再現性確認)。

### 2. Qwen MoE AOT compile None bug

`Qwen3_5MoeForConditionalGeneration` の torch.compile AOT trace 経路で `NoneType has no attribute 'size'`。
workaround は `--compilation-config '{"ir_enable_torch_wrap":false}'` で AOT trace のみ無効化。
本記事の k=2/3/4/5 全部で同じ workaround で起動成功。

## 次の記事 (予定)

本記事は **「各モデルの本機上限値とそれを引き出す knob」** に絞りました。次に書く予定:

- **横断 knob sweep**: `--kv-cache-dtype` (bf16/fp8/nvfp4) × 4 モデル、`--max-num-batched-tokens` 最適値 map、MoE backend 強制切替 (FLASHINFER_TRTLLM / CUTLASS / VLLM_CUTLASS)
- **日本語品質の bench**: `bench-jp.py` 自作で 20-30 個の JP prompt × カテゴリ別 (敬語切替 / 翻訳 / 慣用句 / 技術 docstring 等)、LLM-as-judge で side-by-side。unsloth NVFP4 Qwen MoE の品質劣化が見えるかどうか
- **DFlash drafter on Gemma MoE** (z-lab HF 承認待ち)
- **vLLM 0.21+ で FA4 sm120 解禁** ([Issue #36865](https://github.com/vllm-project/vllm/issues/36865)) を待って Gemma 31B の attention backend を切替

## 参考リンク

- 前記事 (Phase 1+2): <https://zenn.dev/dssugar/articles/rtx5090-vllm-qwen-gemma-2026-05>
- vLLM: <https://github.com/vllm-project/vllm>
- Spec-Bench dataset: <https://github.com/hemingkx/Spec-Bench>
- MTP (Multi-Token Prediction) paper / Qwen 3.6: <https://qwen.ai/blog?id=qwen3.6-27b>
- EAGLE-3 (RedHat speculator series): <https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3>
- 本機の hw 構成: [dssugar/dai-gpu-server (PRIVATE)](https://github.com/dssugar/dai-gpu-server)
