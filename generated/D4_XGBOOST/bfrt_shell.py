#!/usr/bin/env python3
# bfshell bfrt_python control plane — d4  (combined, 2 entries, default=no_deflect)
# Run with: $SDE/run_bfshell.sh -b <this_file>

ig = bfrt.d4.pipe.SwitchIngress

# ── Stage 1: Encoding tables ─────────────────────────────────────────────

# local_queue_occ: 8 buckets
_t = ig.encode_local_queue_occ
_t.clear()
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=0, local_queue_occ_end=-1, bucket=0)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=0, local_queue_occ_end=4137, bucket=1)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=4138, local_queue_occ_end=4636, bucket=2)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=4637, local_queue_occ_end=5605, bucket=3)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=5606, local_queue_occ_end=6019, bucket=4)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=6020, local_queue_occ_end=7927, bucket=5)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=7928, local_queue_occ_end=8443, bucket=6)
_t.add_with_set_bucket_local_queue_occ(local_queue_occ_start=8444, local_queue_occ_end=65535, bucket=7)

# min_cand_occ: 5 buckets
_t = ig.encode_min_cand_occ
_t.clear()
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=0, min_cand_occ_end=-1, bucket=0)
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=0, min_cand_occ_end=4640, bucket=1)
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=4641, min_cand_occ_end=7921, bucket=2)
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=7922, min_cand_occ_end=8567, bucket=3)
_t.add_with_set_bucket_min_cand_occ(min_cand_occ_start=8568, min_cand_occ_end=65535, bucket=4)

# cand2_occ: 6 buckets
_t = ig.encode_cand2_occ
_t.clear()
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=0, cand2_occ_end=13, bucket=0)
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=14, cand2_occ_end=5936, bucket=1)
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=5937, cand2_occ_end=8035, bucket=2)
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=8036, cand2_occ_end=8198, bucket=3)
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=8199, cand2_occ_end=8539, bucket=4)
_t.add_with_set_bucket_cand2_occ(cand2_occ_start=8540, cand2_occ_end=65535, bucket=5)

# cand_occ_spread: 12 buckets
_t = ig.encode_cand_occ_spread
_t.clear()
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=0, cand_occ_spread_end=76, bucket=0)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=77, cand_occ_spread_end=133, bucket=1)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=134, cand_occ_spread_end=2194, bucket=2)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=2195, cand_occ_spread_end=2577, bucket=3)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=2578, cand_occ_spread_end=2677, bucket=4)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=2678, cand_occ_spread_end=3287, bucket=5)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=3288, cand_occ_spread_end=4276, bucket=6)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=4277, cand_occ_spread_end=4415, bucket=7)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=4416, cand_occ_spread_end=4979, bucket=8)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=4980, cand_occ_spread_end=6265, bucket=9)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=6266, cand_occ_spread_end=7378, bucket=10)
_t.add_with_set_bucket_cand_occ_spread(cand_occ_spread_start=7379, cand_occ_spread_end=65535, bucket=11)

# ── Combined decision table (hit = do_deflect, miss = no deflect) ───────────

# model_decision: 2 entries
_t = ig.model_decision
_t.clear()
_t.add_with_do_deflect(bucket_local_queue_occ_start=7, bucket_local_queue_occ_end=7, bucket_min_cand_occ_start=0, bucket_min_cand_occ_end=4, bucket_cand2_occ_start=4, bucket_cand2_occ_end=5, bucket_cand_occ_spread_start=0, bucket_cand_occ_spread_end=4)
_t.add_with_do_deflect(bucket_local_queue_occ_start=7, bucket_local_queue_occ_end=7, bucket_min_cand_occ_start=3, bucket_min_cand_occ_end=4, bucket_cand2_occ_start=4, bucket_cand2_occ_end=5, bucket_cand_occ_spread_start=5, bucket_cand_occ_spread_end=5)

print('All tables populated.')