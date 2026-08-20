# Experiment 1.1: Complete Bdev Loss Atlas

## Scope

Read-only analysis of the retained complete public GOT-10k validation AO JSONs
for Nodes 1, 2, 3, and 6 against the immutable baseline. No tracker was run;
no evaluator/data source was modified; no B_test was accessed.

## Result

| Method | Mean delta | Median delta | Improved / degraded | Top-5 loss share | Worst sequence delta |
| --- | ---: | ---: | ---: | ---: | --- |
| TRM-FAR | -0.008228 | -0.003931 | 70 / 110 | 33.76% | `000042`: -0.339516 |
| Counterfactual agreement | -0.019995 | -0.004659 | 77 / 103 | 35.26% | `000108`: -0.764713 |
| Stable-anchor recovery | -0.032038 | -0.007263 | 66 / 114 | 33.54% | `000108`: -0.766722 |
| Logarithmic sampler | -0.002730 | -0.000534 | 84 / 96 | 55.89% | `000071`: -0.422795 |

The three memory interventions have both negative means and negative medians,
with their five largest losses accounting for about one third of total negative
sequence delta. The logarithmic sampler has a near-zero median and smallest
mean loss, but 55.89% of its total loss arises from five sequences. Thus all
four mechanisms have mixed per-sequence effects; their aggregate failures are
not explained by one shared failure profile. This is descriptive evidence, not
an attribution of semantic causes to individual sequences.

## Reproduction

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
$VENV scripts/analyze_complete_bdev_loss_atlas.py \
  --baseline /root/autodl-tmp/experiment/.research-assets/output/got10k_val_baseline_ao.json \
  --candidate-dir /root/autodl-tmp/experiment/.research-assets/output \
  --output-dir results/1.1-complete-bdev-loss-atlas \
  --bootstrap-samples 10000 --bootstrap-seed 20260820
```

Outputs: `summary.json`, `summary.csv`, and `loss_atlas.png`.
