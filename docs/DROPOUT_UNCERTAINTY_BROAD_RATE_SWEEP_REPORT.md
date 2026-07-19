# Broad MC-Dropout Rate Sweep Report

## Executive Summary

The 500K broad MC-dropout rate sweep shows a monotonic fidelity and classification
degradation as broad dropout increases. The two smallest stochastic rates, `p=1e-5`
and `p=0.001`, retain high agreement with the deterministic full score. At `p=0.01`,
mean uncertainty roughly doubles and the difficult hard-positive versus hard-negative
classification weakens substantially. The repaired `p=0.05` reference continues the
same decline.

The zero-dropout control passed: its color ranking agrees closely with the full score
(Spearman 0.9770), and its top-1/64 selection recalls 0.9960 of the deterministic
reference selection.

## Sweep Metrics

| rate | K | Spearman vs full | mean pairwise AUC | hp_vs_hn AUC | recall@1/64 | mean MC std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.9770 | 0.9717 | 0.8374 | 0.9960 | 0.0000 |
| 1e-5 | 8 | 0.9486 | 0.9368 | 0.6528 | 0.9891 | 0.0080 |
| 0.001 | 8 | 0.9467 | 0.9347 | 0.6428 | 0.9886 | 0.0085 |
| 0.01 | 8 | 0.9196 | 0.9091 | 0.5602 | 0.9738 | 0.0163 |
| 0.05 reference | 8 | 0.8289 | 0.8508 | 0.5172 | 0.9174 | 0.0326 |

`p=1e-5` and `p=0.001` differ by only 0.0019 in Spearman correlation and 0.0005
in recall@1/64. Increasing from `p=0.001` to `p=0.01` decreases Spearman by 0.0271,
mean pairwise AUC by 0.0255, and recall@1/64 by 0.0148. The `p=0.05` result is a
repaired legacy reference and has no directly comparable runtime measurement.

## Pairwise Mean Strategy

Each task compares 100,000 positive rows with 100,000 negative rows. `AUC`, `AP`,
and `balanced F1` use the mean strategy. `F1 tau64` applies the shared fixed cutoff
used by the sweep. Values in parentheses are absolute changes from the `p=0` control
for the same task. The control uses `K=1`, so the controlled rate comparison is among
the nonzero `K=8` rows.

## Per-Task Results Appendix

### `hp_vs_hn`: hard positive vs hard negative

| p | K | AUC (delta) | AP (delta) | balanced F1 (delta) | F1 tau64 (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.8374 (baseline) | 0.8337 (baseline) | 0.7573 (baseline) | 0.7571 (baseline) |
| 1e-5 | 8 | 0.6528 (-0.1846) | 0.6391 (-0.1945) | 0.6121 (-0.1453) | 0.2890 (-0.4681) |
| 0.001 | 8 | 0.6428 (-0.1946) | 0.6288 (-0.2048) | 0.6051 (-0.1522) | 0.2908 (-0.4663) |
| 0.01 | 8 | 0.5602 (-0.2772) | 0.5497 (-0.2840) | 0.5444 (-0.2129) | 0.1864 (-0.5707) |
| 0.05 reference | 8 | 0.5172 (-0.3201) | 0.5137 (-0.3200) | 0.5126 (-0.2447) | 0.3712 (-0.3859) |

### `hp_vs_rn`: hard positive vs random negative

| p | K | AUC (delta) | AP (delta) | balanced F1 (delta) | F1 tau64 (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.9964 (baseline) | 0.9951 (baseline) | 0.9804 (baseline) | 0.8590 (baseline) |
| 1e-5 | 8 | 0.9840 (-0.0124) | 0.9771 (-0.0181) | 0.9465 (-0.0339) | 0.3052 (-0.5539) |
| 0.001 | 8 | 0.9824 (-0.0139) | 0.9748 (-0.0204) | 0.9431 (-0.0374) | 0.3086 (-0.5504) |
| 0.01 | 8 | 0.9445 (-0.0518) | 0.9233 (-0.0718) | 0.8814 (-0.0990) | 0.1984 (-0.6606) |
| 0.05 reference | 8 | 0.7824 (-0.2140) | 0.7596 (-0.2355) | 0.7257 (-0.2547) | 0.4281 (-0.4309) |

### `hp_vs_tn`: hard positive vs tail negative

| p | K | AUC (delta) | AP (delta) | balanced F1 (delta) | F1 tau64 (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.9999 (baseline) | 0.9999 (baseline) | 0.9994 (baseline) | 0.8614 (baseline) |
| 1e-5 | 8 | 0.9998 (-0.0002) | 0.9996 (-0.0003) | 0.9979 (-0.0015) | 0.3057 (-0.5557) |
| 0.001 | 8 | 0.9998 (-0.0002) | 0.9996 (-0.0003) | 0.9978 (-0.0016) | 0.3092 (-0.5522) |
| 0.01 | 8 | 0.9993 (-0.0006) | 0.9991 (-0.0009) | 0.9941 (-0.0054) | 0.1994 (-0.6620) |
| 0.05 reference | 8 | 0.9950 (-0.0049) | 0.9934 (-0.0065) | 0.9722 (-0.0273) | 0.4477 (-0.4136) |

### `rp_vs_hn`: random positive vs hard negative

| p | K | AUC (delta) | AP (delta) | balanced F1 (delta) | F1 tau64 (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.9964 (baseline) | 0.9973 (baseline) | 0.9809 (baseline) | 0.8896 (baseline) |
| 1e-5 | 8 | 0.9851 (-0.0113) | 0.9894 (-0.0078) | 0.9537 (-0.0272) | 0.9464 (+0.0568) |
| 0.001 | 8 | 0.9838 (-0.0126) | 0.9885 (-0.0087) | 0.9510 (-0.0299) | 0.9421 (+0.0525) |
| 0.01 | 8 | 0.9559 (-0.0405) | 0.9686 (-0.0287) | 0.9009 (-0.0800) | 0.9058 (+0.0162) |
| 0.05 reference | 8 | 0.8675 (-0.1289) | 0.9015 (-0.0958) | 0.7934 (-0.1875) | 0.7861 (-0.1035) |

### `rp_vs_rn`: random positive vs random negative

| p | K | AUC (delta) | AP (delta) | balanced F1 (delta) | F1 tau64 (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.9999 (baseline) | 0.9999 (baseline) | 0.9951 (baseline) | 0.9951 (baseline) |
| 1e-5 | 8 | 0.9994 (-0.0006) | 0.9994 (-0.0006) | 0.9863 (-0.0087) | 0.9784 (-0.0167) |
| 0.001 | 8 | 0.9993 (-0.0007) | 0.9993 (-0.0006) | 0.9853 (-0.0098) | 0.9771 (-0.0180) |
| 0.01 | 8 | 0.9949 (-0.0050) | 0.9953 (-0.0046) | 0.9635 (-0.0315) | 0.9401 (-0.0550) |
| 0.05 reference | 8 | 0.9439 (-0.0561) | 0.9559 (-0.0440) | 0.8797 (-0.1154) | 0.8724 (-0.1226) |

### `rp_vs_tn`: random positive vs tail negative

| p | K | AUC (delta) | AP (delta) | balanced F1 (delta) | F1 tau64 (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1.0000 (baseline) | 1.0000 (baseline) | 0.9997 (baseline) | 0.9975 (baseline) |
| 1e-5 | 8 | 1.0000 (-0.0000) | 1.0000 (-0.0000) | 0.9990 (-0.0007) | 0.9793 (-0.0181) |
| 0.001 | 8 | 1.0000 (-0.0000) | 1.0000 (-0.0000) | 0.9990 (-0.0007) | 0.9783 (-0.0192) |
| 0.01 | 8 | 0.9999 (-0.0001) | 0.9999 (-0.0001) | 0.9969 (-0.0028) | 0.9427 (-0.0548) |
| 0.05 reference | 8 | 0.9986 (-0.0014) | 0.9986 (-0.0014) | 0.9839 (-0.0159) | 0.9008 (-0.0967) |

### Alignment-Fixed Variant Context

The following tables retain every result from the alignment-fixed report. The final
row in each table is the repaired `p=0.05` reference already shown above. The
rate-sweep bundle contains the pairwise mean metrics for the new rates, but not
per-task color correlations or mean color shifts; those cells are therefore `--`.

#### `hp_vs_hn`: hard positive vs hard negative

| variant | family | p | K | ROC AUC | AP | F1 original cutoff | F1 balanced rate | Spearman color | mean color shift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full rescore | baseline | -- | -- | 0.8159 | 0.8113 | 0.7436 | 0.7381 | 0.6249 | -0.0001 |
| Full baseline | baseline | -- | -- | 0.8159 | 0.8113 | 0.7436 | 0.7381 | 0.6249 | -0.0001 |
| Paired mid2 | paired | -- | -- | 0.5098 | 0.5073 | 0.3187 | 0.5069 | 0.0215 | 0.0470 |
| Paired top1 | paired | -- | -- | 0.5066 | 0.5041 | 0.5601 | 0.5053 | 0.0134 | -0.0218 |
| Paired mid4 | paired | -- | -- | 0.5063 | 0.5039 | 0.0954 | 0.5040 | 0.0131 | 0.1543 |
| Paired top2 | paired | -- | -- | 0.5047 | 0.5027 | 0.5594 | 0.5023 | 0.0100 | -0.0313 |
| Marginal top1 only | marg_only | -- | -- | 0.5046 | 0.5022 | 0.6667 | 0.5038 | 0.0099 | -0.9231 |
| Marginal top2 only | marg_only | -- | -- | 0.5038 | 0.5026 | 0.6667 | 0.5027 | 0.0086 | -1.5962 |
| Conditional top1 only | cond_only | -- | -- | 0.5012 | 0.4989 | 0.0037 | 0.5016 | 0.0015 | 0.9012 |
| Cond bot2 / marg top2 | cond_bot_marg_top | -- | -- | 0.5007 | 0.5000 | 0.0003 | 0.5004 | 0.0007 | 3.5712 |
| Cond top2 / marg bot2 | cond_top_marg_bot | -- | -- | 0.5006 | 0.4994 | 0.6666 | 0.5012 | 0.0016 | -4.3031 |
| Cond top6 / marg bot6 | cond_top_marg_bot | -- | -- | 0.4994 | 0.4996 | 0.6667 | 0.5000 | -0.0012 | -4.0672 |
| Broad MC-dropout mean | dropout | 0 | 1 | 0.8374 | 0.8337 | 0.7571 | 0.7573 | -- | -- |
| Broad MC-dropout mean | dropout | 1e-5 | 8 | 0.6528 | 0.6391 | 0.2890 | 0.6121 | -- | -- |
| Broad MC-dropout mean | dropout | 0.001 | 8 | 0.6428 | 0.6288 | 0.2908 | 0.6051 | -- | -- |
| Broad MC-dropout mean | dropout | 0.01 | 8 | 0.5602 | 0.5497 | 0.1864 | 0.5444 | -- | -- |
| **Broad MC-dropout mean** | **dropout** | **0.05** | **8** | **0.5172** | **0.5137** | **0.3712** | **0.5126** | **0.0351** | **0.0218** |

#### `hp_vs_rn`: hard positive vs random negative

| variant | family | p | K | ROC AUC | AP | F1 original cutoff | F1 balanced rate | Spearman color | mean color shift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full rescore | baseline | -- | -- | 0.9956 | 0.9939 | 0.8636 | 0.9784 | 0.9181 | -0.0001 |
| Full baseline | baseline | -- | -- | 0.9956 | 0.9939 | 0.8636 | 0.9784 | 0.9181 | -0.0001 |
| Paired mid2 | paired | -- | -- | 0.6792 | 0.6618 | 0.3523 | 0.6378 | 0.3449 | 0.0469 |
| Paired top1 | paired | -- | -- | 0.6189 | 0.5975 | 0.6047 | 0.5877 | 0.2270 | -0.0244 |
| Paired mid4 | paired | -- | -- | 0.6116 | 0.5929 | 0.0975 | 0.5841 | 0.2174 | 0.1530 |
| Paired top2 | paired | -- | -- | 0.5896 | 0.5691 | 0.5915 | 0.5653 | 0.1715 | -0.0350 |
| Marginal top1 only | marg_only | -- | -- | 0.5741 | 0.5529 | 0.6667 | 0.5562 | 0.1435 | -0.9198 |
| Marginal top2 only | marg_only | -- | -- | 0.5585 | 0.5449 | 0.6667 | 0.5436 | 0.1126 | -1.5907 |
| Conditional top1 only | cond_only | -- | -- | 0.5312 | 0.5262 | 0.0037 | 0.5229 | 0.0588 | 0.8952 |
| Cond bot2 / marg top2 | cond_bot_marg_top | -- | -- | 0.5016 | 0.5020 | 0.0003 | 0.5029 | 0.0037 | 3.5268 |
| Cond top2 / marg bot2 | cond_top_marg_bot | -- | -- | 0.5224 | 0.5207 | 0.6667 | 0.5172 | 0.0425 | -4.2731 |
| Cond top6 / marg bot6 | cond_top_marg_bot | -- | -- | 0.4931 | 0.4952 | 0.6667 | 0.4954 | -0.0143 | -4.0905 |
| Broad MC-dropout mean | dropout | 0 | 1 | 0.9964 | 0.9951 | 0.8590 | 0.9804 | -- | -- |
| Broad MC-dropout mean | dropout | 1e-5 | 8 | 0.9840 | 0.9771 | 0.3052 | 0.9465 | -- | -- |
| Broad MC-dropout mean | dropout | 0.001 | 8 | 0.9824 | 0.9748 | 0.3086 | 0.9431 | -- | -- |
| Broad MC-dropout mean | dropout | 0.01 | 8 | 0.9445 | 0.9233 | 0.1984 | 0.8814 | -- | -- |
| **Broad MC-dropout mean** | **dropout** | **0.05** | **8** | **0.7824** | **0.7596** | **0.4281** | **0.7257** | **0.5387** | **0.0232** |

#### `rp_vs_hn`: random positive vs hard negative

| variant | family | p | K | ROC AUC | AP | F1 original cutoff | F1 balanced rate | Spearman color | mean color shift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full rescore | baseline | -- | -- | 0.9958 | 0.9969 | 0.8703 | 0.9796 | 0.9176 | -0.0001 |
| Full baseline | baseline | -- | -- | 0.9958 | 0.9969 | 0.8703 | 0.9796 | 0.9176 | -0.0001 |
| Paired mid2 | paired | -- | -- | 0.8013 | 0.8430 | 0.7235 | 0.7269 | 0.6277 | 0.0469 |
| Paired top1 | paired | -- | -- | 0.7605 | 0.7984 | 0.6952 | 0.6885 | 0.5551 | -0.0191 |
| Paired mid4 | paired | -- | -- | 0.7426 | 0.7735 | 0.5126 | 0.6791 | 0.5222 | 0.1569 |
| Paired top2 | paired | -- | -- | 0.7260 | 0.7534 | 0.6857 | 0.6614 | 0.4912 | -0.0234 |
| Marginal top1 only | marg_only | -- | -- | 0.6909 | 0.7012 | 0.6667 | 0.6437 | 0.4218 | -0.9400 |
| Marginal top2 only | marg_only | -- | -- | 0.6692 | 0.6772 | 0.6667 | 0.6276 | 0.3766 | -1.6256 |
| Conditional top1 only | cond_only | -- | -- | 0.6145 | 0.6040 | 0.0102 | 0.5825 | 0.2637 | 0.9208 |
| Cond bot2 / marg top2 | cond_bot_marg_top | -- | -- | 0.5710 | 0.5682 | 0.0003 | 0.5538 | 0.1825 | 3.5796 |
| Cond top2 / marg bot2 | cond_top_marg_bot | -- | -- | 0.4983 | 0.5035 | 0.6666 | 0.4977 | -0.0263 | -4.2621 |
| Cond top6 / marg bot6 | cond_top_marg_bot | -- | -- | 0.4443 | 0.4643 | 0.6667 | 0.4609 | -0.1268 | -3.9588 |
| Broad MC-dropout mean | dropout | 0 | 1 | 0.9964 | 0.9973 | 0.8896 | 0.9809 | -- | -- |
| Broad MC-dropout mean | dropout | 1e-5 | 8 | 0.9851 | 0.9894 | 0.9464 | 0.9537 | -- | -- |
| Broad MC-dropout mean | dropout | 0.001 | 8 | 0.9838 | 0.9885 | 0.9421 | 0.9510 | -- | -- |
| Broad MC-dropout mean | dropout | 0.01 | 8 | 0.9559 | 0.9686 | 0.9058 | 0.9009 | -- | -- |
| **Broad MC-dropout mean** | **dropout** | **0.05** | **8** | **0.8675** | **0.9015** | **0.7861** | **0.7934** | **0.7285** | **0.0184** |

#### `rp_vs_rn`: random positive vs random negative

| variant | family | p | K | ROC AUC | AP | F1 original cutoff | F1 balanced rate | Spearman color | mean color shift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full rescore | baseline | -- | -- | 0.9999 | 0.9999 | 0.9945 | 0.9946 | 0.9996 | -0.0001 |
| Full baseline | baseline | -- | -- | 0.9999 | 0.9999 | 0.9945 | 0.9946 | 0.9996 | -0.0001 |
| Paired mid2 | paired | -- | -- | 0.8780 | 0.9021 | 0.7800 | 0.8033 | 0.7669 | 0.0468 |
| Paired top1 | paired | -- | -- | 0.8254 | 0.8493 | 0.7450 | 0.7446 | 0.6708 | -0.0218 |
| Paired mid4 | paired | -- | -- | 0.8070 | 0.8293 | 0.5212 | 0.7357 | 0.6407 | 0.1555 |
| Paired top2 | paired | -- | -- | 0.7825 | 0.8003 | 0.7215 | 0.7093 | 0.5925 | -0.0272 |
| Marginal top1 only | marg_only | -- | -- | 0.7396 | 0.7425 | 0.6667 | 0.6848 | 0.5106 | -0.9367 |
| Marginal top2 only | marg_only | -- | -- | 0.7117 | 0.7157 | 0.6667 | 0.6618 | 0.4539 | -1.6201 |
| Conditional top1 only | cond_only | -- | -- | 0.6421 | 0.6365 | 0.0102 | 0.6025 | 0.3154 | 0.9149 |
| Cond bot2 / marg top2 | cond_bot_marg_top | -- | -- | 0.5723 | 0.5701 | 0.0003 | 0.5568 | 0.1863 | 3.5352 |
| Cond top2 / marg bot2 | cond_top_marg_bot | -- | -- | 0.5202 | 0.5241 | 0.6667 | 0.5139 | 0.0149 | -4.2321 |
| Cond top6 / marg bot6 | cond_top_marg_bot | -- | -- | 0.4382 | 0.4605 | 0.6667 | 0.4559 | -0.1396 | -3.9821 |
| Broad MC-dropout mean | dropout | 0 | 1 | 0.9999 | 0.9999 | 0.9951 | 0.9951 | -- | -- |
| Broad MC-dropout mean | dropout | 1e-5 | 8 | 0.9994 | 0.9994 | 0.9784 | 0.9863 | -- | -- |
| Broad MC-dropout mean | dropout | 0.001 | 8 | 0.9993 | 0.9993 | 0.9771 | 0.9853 | -- | -- |
| Broad MC-dropout mean | dropout | 0.01 | 8 | 0.9949 | 0.9953 | 0.9401 | 0.9635 | -- | -- |
| **Broad MC-dropout mean** | **dropout** | **0.05** | **8** | **0.9439** | **0.9559** | **0.8724** | **0.8797** | **0.8813** | **0.0198** |

#### `hp_vs_tn`: hard positive vs tail negative

| variant | family | p | K | ROC AUC | AP | F1 original cutoff | F1 balanced rate | Spearman color | mean color shift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full rescore | baseline | -- | -- | 0.9999 | 0.9999 | 0.8666 | 0.9994 | 0.9210 | -0.0002 |
| Full baseline | baseline | -- | -- | 0.9999 | 0.9999 | 0.8666 | 0.9994 | 0.9210 | -0.0002 |
| Paired mid2 | paired | -- | -- | 0.9852 | 0.9826 | 0.3755 | 0.9513 | 0.8506 | 0.0578 |
| Paired top1 | paired | -- | -- | 0.9571 | 0.9475 | 0.7587 | 0.9016 | 0.8012 | -0.0779 |
| Paired mid4 | paired | -- | -- | 0.9588 | 0.9520 | 0.1000 | 0.9108 | 0.7983 | 0.1375 |
| Paired top2 | paired | -- | -- | 0.8942 | 0.8693 | 0.7306 | 0.8191 | 0.6879 | -0.1217 |
| Marginal top1 only | marg_only | -- | -- | 0.9136 | 0.8933 | 0.6727 | 0.8542 | 0.7364 | -0.9097 |
| Marginal top2 only | marg_only | -- | -- | 0.8319 | 0.8154 | 0.6669 | 0.7649 | 0.5853 | -1.6176 |
| Conditional top1 only | cond_only | -- | -- | 0.8052 | 0.7956 | 0.0037 | 0.7353 | 0.5776 | 0.8317 |
| Cond bot2 / marg top2 | cond_bot_marg_top | -- | -- | 0.7075 | 0.7122 | 0.0003 | 0.6774 | 0.4018 | 3.5354 |
| Cond top2 / marg bot2 | cond_top_marg_bot | -- | -- | 0.4723 | 0.5209 | 0.6666 | 0.4733 | -0.0827 | -4.4517 |
| Cond top6 / marg bot6 | cond_top_marg_bot | -- | -- | 0.4543 | 0.4937 | 0.6667 | 0.4647 | -0.0909 | -4.3028 |
| Broad MC-dropout mean | dropout | 0 | 1 | 0.9999 | 0.9999 | 0.8614 | 0.9994 | -- | -- |
| Broad MC-dropout mean | dropout | 1e-5 | 8 | 0.9998 | 0.9996 | 0.3057 | 0.9979 | -- | -- |
| Broad MC-dropout mean | dropout | 0.001 | 8 | 0.9998 | 0.9996 | 0.3092 | 0.9978 | -- | -- |
| Broad MC-dropout mean | dropout | 0.01 | 8 | 0.9993 | 0.9991 | 0.1994 | 0.9941 | -- | -- |
| **Broad MC-dropout mean** | **dropout** | **0.05** | **8** | **0.9950** | **0.9934** | **0.4477** | **0.9722** | **0.8682** | **0.0395** |

#### `rp_vs_tn`: random positive vs tail negative

| variant | family | p | K | ROC AUC | AP | F1 original cutoff | F1 balanced rate | Spearman color | mean color shift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full rescore | baseline | -- | -- | 1.0000 | 1.0000 | 0.9975 | 0.9997 | 1.0000 | -0.0001 |
| Full baseline | baseline | -- | -- | 1.0000 | 1.0000 | 0.9975 | 0.9997 | 1.0000 | -0.0001 |
| Paired mid2 | paired | -- | -- | 0.9936 | 0.9941 | 0.8174 | 0.9675 | 0.9607 | 0.0577 |
| Paired top1 | paired | -- | -- | 0.9821 | 0.9824 | 0.9114 | 0.9373 | 0.9238 | -0.0752 |
| Paired mid4 | paired | -- | -- | 0.9787 | 0.9803 | 0.5315 | 0.9391 | 0.9103 | 0.1400 |
| Paired top2 | paired | -- | -- | 0.9502 | 0.9488 | 0.8722 | 0.8811 | 0.8534 | -0.1139 |
| Marginal top1 only | marg_only | -- | -- | 0.9465 | 0.9457 | 0.6727 | 0.8930 | 0.8552 | -0.9266 |
| Marginal top2 only | marg_only | -- | -- | 0.8935 | 0.9000 | 0.6669 | 0.8309 | 0.7476 | -1.6470 |
| Conditional top1 only | cond_only | -- | -- | 0.8646 | 0.8670 | 0.0102 | 0.7891 | 0.7223 | 0.8513 |
| Cond bot2 / marg top2 | cond_bot_marg_top | -- | -- | 0.7516 | 0.7696 | 0.0003 | 0.7137 | 0.5260 | 3.5438 |
| Cond top2 / marg bot2 | cond_top_marg_bot | -- | -- | 0.4689 | 0.5205 | 0.6666 | 0.4692 | -0.1138 | -4.4107 |
| Cond top6 / marg bot6 | cond_top_marg_bot | -- | -- | 0.3951 | 0.4534 | 0.6667 | 0.4212 | -0.2245 | -4.1944 |
| Broad MC-dropout mean | dropout | 0 | 1 | 1.0000 | 1.0000 | 0.9975 | 0.9997 | -- | -- |
| Broad MC-dropout mean | dropout | 1e-5 | 8 | 1.0000 | 1.0000 | 0.9793 | 0.9990 | -- | -- |
| Broad MC-dropout mean | dropout | 0.001 | 8 | 1.0000 | 1.0000 | 0.9783 | 0.9990 | -- | -- |
| Broad MC-dropout mean | dropout | 0.01 | 8 | 0.9999 | 0.9999 | 0.9427 | 0.9969 | -- | -- |
| **Broad MC-dropout mean** | **dropout** | **0.05** | **8** | **0.9986** | **0.9986** | **0.9008** | **0.9839** | **0.9830** | **0.0361** |

## Use-Case Conclusions

| use case | verdict | evidence |
| --- | --- | --- |
| broad stochastic scoring at `p <= 0.001` | supported for high-fidelity ranking | Spearman remains at least 0.9467 and recall@1/64 at least 0.9886. |
| broad stochastic scoring at `p=0.01` | usable only with a fidelity trade-off | Mean MC std doubles from `p=0.001`; difficult-task AUC falls to 0.5602. |
| broad stochastic scoring at `p=0.05` | not preferred for deterministic-ranking fidelity | Spearman 0.8289 and recall@1/64 0.9174. |
| shared fixed threshold | not rate-robust | `F1 tau64` changes sharply with rate even when rank AUC remains near one on easy tasks. |

## Limitations

- The zero-dropout control uses `K=1`; nonzero-rate comparisons use `K=8`.
- One seed and one broad-dropout scope were evaluated.
- The `p=0.05` row is an alignment-repaired legacy reference rather than a new
  production run, and its runtime is not comparable to the four new configurations.
- AUC and AP characterize ranking; fixed-cutoff F1 also depends on score-scale drift.

## Reproducibility

- Rows: 500,000.
- Producer revision: `7d19d836bc48a6ca76621558d5a339ad030af284`.
- Analysis revision: `afe9db7b62bf17dab38ee4a51395e40adcb2dfea`.
- Notebook revision: `broad-rate-sweep-v7-2026-07-18`.
- Acceptance: all sweep configurations complete; zero-dropout control passed;
  corrected `p=0.05` reference included.
- Source bundle: `/Users/myazdani/Downloads/dropout_uncertainty_broad_rate_sweep_bundle`.
