# Expanded Rock-Paper-Scissors — Cornell CS 3/5780 ML Competition

**2nd of ~300 students** (undergraduate and graduate), Spring 2025 Machine Learning for Intelligent Systems. Final private-leaderboard accuracy: **0.9102**, up from a 0.64 linear baseline across 17 submissions.

---

## The problem

Given two 24×24 grayscale images, predict whether the contents of the first beat the contents of the second: `+1` if it wins, `-1` otherwise.

The class labels are never observed. Nothing tells you that a particular image is a rock or a pair of scissors — you see only the outcome of a matchup. The model has to learn "what gesture is this" as an implicit byproduct of learning "which of these two wins," from a binary signal about the pair.

Two structural properties of the target drove every design decision:

**The relation is antisymmetric.** `f(A, B) = -f(B, A)` holds identically, by definition of the task. A hard constraint known in advance, not something to be estimated.

**The relation is non-transitive.** Rock beats scissors beats paper beats rock. This rules out an entire family of otherwise-natural approaches: any model that maps each image to a scalar and compares the two scalars cannot represent cyclic dominance, because ordering on the real line is transitive and this relation is not. The comparison has to happen between full embedding vectors with enough dimensionality to encode a cycle.

That second property is why the first eleven submissions went nowhere.

## The arc

| Stage | Private | Public |
|---|---|---|
| Linear regression, 11 iterations | 0.559 → 0.644 | 0.553 → 0.631 |
| Early CNN attempts | 0.761 → 0.769 | 0.755 → 0.769 |
| Siamese ResNet-18, single model | 0.876 | 0.881 |
| Tuning, more epochs | 0.891 | 0.890 |
| Siamese ResNet-50, 5-model bagged ensemble | 0.9066 | 0.9073 |
| + `\|f1−f2\|` block, stratified subsampling | 0.9080 | 0.9066 |
| **Final submission** | **0.9102** | 0.9084 |

Seventeen submissions, two of which errored out.

### The two changes that mattered

**Linear regression → Siamese CNN: +0.23.** Everything before this was capped in the low 0.6s, and in hindsight that ceiling was structural rather than a tuning problem. A linear model on raw pixels induces an ordering, and an ordering cannot express rock-paper-scissors. Eleven submissions of parameter tweaking bought roughly nine points; changing hypothesis class bought twenty-three.

**Single model → 5-model bagged ensemble: +0.016.** Five ResNet-50s each trained on a different 80% subsample, sigmoid outputs averaged. Each model overfits its own subsample differently, and averaging cancels a large part of that variance. The competition warned explicitly that public-leaderboard standing wouldn't predict private-leaderboard standing, which made variance reduction in the final predictor worth more than another point of single-model fit.

Everything after that was under a point.

## Architecture

Both images pass through a **shared** convolutional encoder; the two embeddings are concatenated and fed to a small MLP head producing one logit, trained with `BCEWithLogitsLoss` against ±1 labels remapped to {0, 1}.

Weight sharing is the key structural choice. Separate encoders per position would learn the same visual features twice from half the data each. Sharing means every training pair contributes two gradient signals to one set of filters.

Images were upscaled 24×24 → 128×128 and replicated to three channels to make ImageNet-pretrained weights usable. With a small training set and an indirect supervision signal, transfer learning was worth more than the resolution mismatch cost.

| | |
|---|---|
| Encoder | ResNet-50, ImageNet-pretrained |
| Head | `[f1, f2]` → 512 → 1, ReLU, dropout 0.3 |
| Optimizer | Adam, lr = 1e-4 |
| Loss | `BCEWithLogitsLoss` |
| Batch size | 64 |
| Epochs | 5 per model |
| Precision | Mixed (AMP) with gradient scaling |
| Hardware | Single Colab GPU |

## What I got wrong

**The `|f1−f2|` block did nothing, and I thought it was the breakthrough.** Adding it moved private accuracy from 0.9066 to 0.9080 — inside the noise on a test set this size. The reason is structural: `|f1−f2|` is symmetric, identical for `(A, B)` and `(B, A)`, so those dimensions carry no information about ordering, and ordering is the only thing the label depends on. It's the standard head for similarity tasks — "are these the same person?" — where symmetry is exactly what you want, and I imported it without checking whether the symmetry assumption matched this problem. It didn't. The signed difference `f1 − f2` flips sign under swap and was the right choice.

**The folds were bagging, not cross-validation.** The loop trained on each subsample and went straight to test prediction; the validation index was never scored. Bagging is a real technique and it delivered the second-largest gain here, but it left no offline model-selection signal, so every architectural decision was judged against public-leaderboard feedback — precisely the thing the competition warned about. Scoring the held-out folds would have cost nothing and would have shown me the `|f1−f2|` block wasn't earning its 2048 dimensions.

**Swap augmentation was free and unused.** Since `f(A, B) = -f(B, A)` by construction, every training pair yields a second valid pair: swap the images, flip the label. Exact doubling of the training set with none of the distributional assumptions that flips or crops require. Very likely the largest remaining improvement, and it costs three lines.

**No test-time symmetrization.** Predicting on both orderings and combining `p(A,B)` with `1 − p(B,A)` enforces antisymmetry at inference for one extra forward pass and no retraining.

**ResNet-50 was probably oversized.** A 24×24 image holds 576 pixels. ResNet-50's stem — 7×7 stride-2 convolution then max pooling — discards a large fraction of that before the first residual block. A small CNN trained from scratch at native resolution is a serious alternative I never tested.

## Repository contents

This repository documents methodology and results. Solution code is not published: the competition carried extra credit toward a course grade and is governed by [Cornell's academic integrity policy](https://www.cs.cornell.edu/courses/cs5780/2025sp/#Policies).

The dataset is not included and is not mine to redistribute.
