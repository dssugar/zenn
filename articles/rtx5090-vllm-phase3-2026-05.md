---
title: "RTX 5090 (32GB) で 30B 級 LLM を vLLM 比較 (Phase 3): longer ctx, spec_bench acceptance, MTP k sweep"
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

本記事はその宿題を回収する Phase 3 の中間記録です。
具体的には **`vllm bench serve` で 3 軸の負荷を掛けて再比較**します:

1. **longer context** — `random` dataset で input 1024 / 4096 token
2. **concurrency** — c=1 vs c=4 で aggregate throughput
3. **natural language acceptance** — `spec_bench` dataset (coding / qa / summarization)
4. **MTP num_speculative_tokens の k sweep** — Qwen MoE で k=2/3/4

結果として、**短文では見えなかった 5 つの発見**が出てきました。

:::message alert
本記事は速度数値だけを扱います。**JP 出力品質 / 並列負荷上限 / 安定性の検証は別記事**で扱う予定で、
**本記事の数値だけで実運用切替の判断はしない**方針です。
unsloth NVFP4 quant の精度劣化、Qwen 系の thinking mode、MoE の並列スケーリング上限など、
速度 ≠ 採用判断であることに注意してください。
:::

## TL;DR

| Workload | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| 短文 chat (Phase 1+2) | Qwen MoE 240 | Gemma MoE 163 | Gemma 31B EAGLE-3 55 | Qwen 27B 53 |
| `random` 1k single | **Qwen MoE+MTP 176** | Gemma MoE 149 | Gemma 31B EAGLE-3 53 | Qwen 27B 50 |
| `random` 4k single | **Qwen MoE+MTP 149** | Gemma MoE 138 | Gemma 31B EAGLE-3 48 | Qwen 27B 47 |
| `random` 1k c=4 aggr | **Gemma MoE 443** | Qwen MoE+MTP 337 | Qwen 27B 170 | Gemma 31B 151 |
| `spec_bench` coding | **Qwen MoE+k4 251** | Gemma MoE 154 | Gemma 31B EAGLE-3 126 | Qwen 27B 51 |
| `spec_bench` qa | **Qwen MoE+k4 272** | Gemma MoE 154 | Gemma 31B EAGLE-3 93 | Qwen 27B 52 |
| `spec_bench` summ | **Qwen MoE+k4 240** | Gemma MoE 149 | Gemma 31B EAGLE-3 80 | Qwen 27B 51 |

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

## 結果 4: Qwen MoE MTP k sweep

`spec_bench` 3 categories × Qwen MoE で `num_speculative_tokens = {2, 3, 4}`:

| k | coding tok/s | qa tok/s | summ tok/s | coding acc len | per-position coding |
|---|---|---|---|---|---|
| 2 | 246.0 | 239.6 | 222.5 | 2.55 | P0 85% / P1 70% |
| 3 | 245.8 | 259.7 | 235.0 | 3.08 | P0 82% / P1 69% / P2 58% |
| **4** | **251.0** | **272.0** | **239.5** | **3.49** | P0 82% / P1 68% / P2 54% / **P3 45%** |

**k=4 が全カテゴリで勝者**です。qa で **272 tok/s** (Phase 1+2 の short-prompt 240 から +13%)。
**P3 が 45% hit** している事実から、k=5 にもまだ余地はあります。
ただし marginal gain は縮小傾向 (k=2→3 で +20 tok/s、k=3→4 で +12 tok/s) なので本記事は k=4 で打ち止め。

これが 5 つ目の発見:

> **Qwen MoE の MTP は k=4 まで上げて勝ち続ける。Phase 1+2 の k=2 は最適ではなかった。**

(他モデルでは MTP / EAGLE-3 の k 増加は逆効果が出やすいので、これも Qwen MoE の MTP 設計の特異性。)

## Qwen 27B (hybrid SSM) は 4k では SSM benefit 出ず

| Model | rand1k c1 | rand4k c1 |
|---|---|---|
| Qwen 27B (hybrid SSM, 64 層中 48 SSM) | 50.2 | 47.4 |
| Gemma 31B (dense) | 52.5 | 48.3 |

hybrid SSM の謳い文句 (long-ctx KV cache が dense より遥かに小さい、262k native) は、
4k input ではほぼ恩恵が見えません。**8k+ で再検証必要** (本記事 deferred)。

仮説: 4k 程度の入力では prefill cost が full attention 16 層分で決まる (dense とほぼ同等)、
SSM 層は KV cache を持たない代わりに固有の per-step 計算がある、結果として速度差が出にくい。

## まとめ: workload × モデル の使い分け候補 (実投入は別記事の品質検証後)

| Workload | 候補 | 数値根拠 (本記事) | 確認必要事項 |
|---|---|---|---|
| 短文 chat / 単発 | Qwen MoE + MTP k=4 | qa 272 tok/s | unsloth NVFP4 quant の JP 品質 |
| 並列負荷 (c=4) | Gemma MoE (no spec) | c=4 で 442 tok/s aggregate | DFlash drafter (HF 承認待ち) で更に伸びるか |
| 現常用 (Gemma 31B + EAGLE-3) | 退役候補 | 全シナリオ 3 番手以下 | EAGLE-3 自体は安定動作の reference |

:::message
**速度数値だけで採用判断はしません**。次の記事 (Phase 3 part 2) で:

- `bench-jp.py` 自作 + 20-30 個の JP prompt 別 LLM-as-judge
- Qwen MoE 8k/16k input 安定性
- Gemma MoE concurrency 8/16 上限

を確認してから実運用切替を決定する予定です。
:::

## 上流 PR 候補 (本記事で発見した 2 件)

### 1. vLLM 0.20.1 SpecBench クラスの self バグ (本記事)

`vllm/benchmarks/datasets/datasets.py:2367` の `def sample(**kwargs)` に `self` 引数追加。
patch は本記事の「落とし穴 1」参照。

### 2. Qwen MoE AOT compile None bug (前記事 Phase 1+2 から継続)

`Qwen3_5MoeForConditionalGeneration` の torch.compile AOT trace 経路で `NoneType has no attribute 'size'`。
workaround は `--compilation-config '{"ir_enable_torch_wrap":false}'` で AOT trace のみ無効化。
本記事の k=2/3/4 全部で同じ workaround で起動成功 (再現性確認)。

## 参考リンク

- 前記事 (Phase 1+2): <https://zenn.dev/dssugar/articles/rtx5090-vllm-qwen-gemma-2026-05>
- vLLM: <https://github.com/vllm-project/vllm>
- Spec-Bench dataset: <https://github.com/hemingkx/Spec-Bench>
- MTP (Multi-Token Prediction) paper / Qwen 3.6: <https://qwen.ai/blog?id=qwen3.6-27b>
- EAGLE-3 (RedHat speculator series): <https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3>
- 本機の hw 構成: [dssugar/dai-gpu-server (PRIVATE)](https://github.com/dssugar/dai-gpu-server)
