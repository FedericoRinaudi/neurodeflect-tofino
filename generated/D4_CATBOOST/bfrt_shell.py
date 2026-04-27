#!/usr/bin/env python3
# bfshell bfrt_python control plane — d4  (combined, 3 entries, default=no_deflect)
# Run with: $SDE/run_bfshell.sh -b <this_file>

ig = bfrt.d4.pipe.SwitchIngress

# ── Stage 1: Encoding tables ─────────────────────────────────────────────

# local_queue_occ: 4 buckets
_t = ig.encode_local_queue_occ
_t.clear()
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=0, local_queue_occ_end=469088, bucket=0)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=469089, local_queue_occ_end=817930, bucket=1)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=817931, local_queue_occ_end=915828, bucket=2)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=915829, local_queue_occ_end=65535, bucket=3)

# min_cand_occ: 2 buckets
_t = ig.encode_min_cand_occ
_t.clear()
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=0, min_cand_occ_end=340429, bucket=0)
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=340430, min_cand_occ_end=65535, bucket=1)

# cand2_occ: 1 buckets
_t = ig.encode_cand2_occ
_t.clear()
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=0, cand2_occ_end=65535, bucket=0)

# cand_occ_spread: 3 buckets
_t = ig.encode_cand_occ_spread
_t.clear()
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=0, cand_occ_spread_end=292526, bucket=0)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=292527, cand_occ_spread_end=381516, bucket=1)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=381517, cand_occ_spread_end=65535, bucket=2)

# occ_vs_min_cand: 3 buckets
_t = ig.encode_occ_vs_min_cand
_t.clear()
_t.add_with_set_bucket_occ_vs_min_cand(occ_vs_start=0, occ_vs_end=953255, bucket=0)
_t.add_with_set_bucket_occ_vs_min_cand(occ_vs_start=953256, occ_vs_end=955677, bucket=1)
_t.add_with_set_bucket_occ_vs_min_cand(occ_vs_start=955678, occ_vs_end=65535, bucket=2)

# occ_vs_mean_cand: 1 buckets
_t = ig.encode_occ_vs_mean_cand
_t.clear()
_t.add_with_set_bucket_occ_vs_mean_cand(occ_vs_start=0, occ_vs_end=65535, bucket=0)

# ── Combined decision table (hit = do_deflect, miss = no deflect) ───────────

# model_decision: 3 entries
_t = ig.model_decision
_t.clear()
_t.add_with_do_deflect(bucket_local_queue_occ_start=2, bucket_local_queue_occ_end=3, bucket_min_cand_occ_start=0, bucket_min_cand_occ_end=0, bucket_cand_occ_spread_start=0, bucket_cand_occ_spread_end=0, bucket_occ_vs_min_cand_start=0, bucket_occ_vs_min_cand_end=2)
_t.add_with_do_deflect(bucket_local_queue_occ_start=3, bucket_local_queue_occ_end=3, bucket_min_cand_occ_start=1, bucket_min_cand_occ_end=1, bucket_cand_occ_spread_start=0, bucket_cand_occ_spread_end=1, bucket_occ_vs_min_cand_start=0, bucket_occ_vs_min_cand_end=2)
_t.add_with_do_deflect(bucket_local_queue_occ_start=2, bucket_local_queue_occ_end=3, bucket_min_cand_occ_start=0, bucket_min_cand_occ_end=0, bucket_cand_occ_spread_start=1, bucket_cand_occ_spread_end=2, bucket_occ_vs_min_cand_start=0, bucket_occ_vs_min_cand_end=1)

print('All tables populated.')