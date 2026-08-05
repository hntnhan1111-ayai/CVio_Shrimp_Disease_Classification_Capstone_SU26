# ASL-LDAM explained

## What the implementation computes

ASL-LDAM is the repository's composition of a label-distribution-aware margin
with a single-label adaptation of asymmetric loss. It targets two separate
imbalances: unequal class representation and unequal optimization pressure from
positive versus negative class terms.

![Flow diagram in which class-count margins modify the true-class logit before single-label asymmetric loss](assets/diagrams/asl_ldam_flow.svg)

*Figure 1. Exact implementation order: LDAM adjustment → scale → single-label
ASL. This order is a project design choice, not a theorem of optimality.*

## Definitions

- A **logit** `z_j` is the unnormalized score for class `j`.
- The **true-class logit** `z_y` is the score assigned to the ground-truth class
  `y` for the current image.
- A **class count** `n_j` is the number of training images in class `j`.
- A **class frequency** is that count divided by the total training count.
- A **margin** is an amount subtracted from the true-class logit during training,
  making the decision requirement stricter. Rarer classes receive larger LDAM
  margins in this implementation.
- An **easy negative** is an incorrect class to which the model already assigns
  very low probability. A **hard negative** is an incorrect class that still
  receives substantial probability and therefore competes with the true class.

## LDAM step

For class `j`, the unnormalized margin is

```text
r_j = 1 / n_j^(1/4)
```

and the repository scales it so that the largest margin equals `max_m`:

```text
m_j = r_j * max_m / max_k(r_k)
```

Only the true-class logit is changed:

```text
z'_j = scale * (z_j - 1[j = y] * m_y)
```

The retained SDI-4 configuration uses training class counts
`[282, 139, 229, 154]`, `max_m=0.5`, and `scale=30.0`.

## Single-label ASL step

The implementation computes `p = softmax(z')`, one-hot targets, and a focusing
weight per class. With `gamma_pos=0` and `gamma_neg=4`, correctly suppressed
negative classes receive much less weight than confusing negative classes.
Label smoothing of `0.1` is then applied to the target distribution before the
weighted negative log-likelihood is summed across classes and averaged across
the batch.

This differs from simply applying the original multi-label sigmoid ASL formula:
the repository uses mutually exclusive classes and softmax. The adaptation is
visible in [`asl_ldam.py`](../src/cvio_asl_ldam/losses/asl_ldam.py).

## Scope of the claim

LDAM was introduced for imbalanced classification by
[Cao et al.](https://proceedings.nips.cc/paper_files/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html),
and ASL by [Ridnik et al.](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html).
The particular **LDAM → single-label ASL** implementation order used here has
not been proven optimal against all alternative orderings, logits/probability
formulations, reweighting schemes, or hyperparameter choices.
