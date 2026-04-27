#include <core.p4>
#include <tna.p4>

#include "includes/defines.p4"
#include "includes/headers.p4"
#include "includes/parser.p4"

/*
    Switch Ingress Pipeline
*/

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

    // --- Actions ---

    action drop() {
        ig_intr_dprsr_md.drop_ctl = 1;
    }

    action set_deflect_port(bit<16> port) {
        ig_intr_tm_md.ucast_egress_port = (bit<9>)port;
    }

    action nop() { }

    // Standard L3 Forwarding
    action ipv4_forward(bit<16> port, bit<2> port_logical, bit<48> dstAddr, bit<48> srcAddr) {
        ig_intr_tm_md.ucast_egress_port = (bit<9>)port;
        ig_md.port_idx = port_logical;
        hdr.ethernet.srcAddr = srcAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }


    #define DECLARE_IG_QUEUE_LEN_REG(N)                                        \
        Register<bit<16>, bit<1>>(1) ig_queue##N##_len_reg;                   \
        RegisterAction<bit<16>, bit<1>, bit<16>>(ig_queue##N##_len_reg)        \
        r_ig_queue##N##_len_alu = {                                            \
            void apply(inout bit<16> register_data, out bit<16> result) {      \
                result = register_data;                                         \
            }                                                                   \
        };                                                                     \
        RegisterAction<bit<16>, bit<1>, void>(ig_queue##N##_len_reg)           \
        w_ig_queue##N##_len_alu = {                                            \
            void apply(inout bit<16> register_data) {                          \
                register_data = (bit<16>)(hdr.bee.queue_occ);                  \
            }                                                                   \
        };                                                                       \
        action read_ig_queue##N##_len() {                                               \
            ig_md.q##N##_occ = r_ig_queue##N##_len_alu.execute(0);              \
        }                                                                        \
        action write_ig_queue##N##_len() {                                              \
            w_ig_queue##N##_len_alu.execute(0);                                              \
        }                                                                          \
        table write_ig_queue##N##_len_tbl {                                                   \
            actions = { write_ig_queue##N##_len;}                                   \
            default_action = write_ig_queue##N##_len();                                                        \
            size = 1;                                                                      \
        }                                                                         \
        table read_ig_queue##N##_len_tbl {                                                   \
            actions = { read_ig_queue##N##_len;}                                   \
            default_action = read_ig_queue##N##_len();                                                        \
            size = 1;                                                                      \
        }

    DECLARE_IG_QUEUE_LEN_REG(0)
    DECLARE_IG_QUEUE_LEN_REG(1)
    DECLARE_IG_QUEUE_LEN_REG(2)
    DECLARE_IG_QUEUE_LEN_REG(3)

    // 1. Forwarding Table
    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr : lpm;
        }
        actions = {
            ipv4_forward;
            //manage_bee;
            drop;
        }
        size = TABLE_SIZE;
        default_action = drop();
    }

    // 2. Convert logical deflection port to physical
    table deflect_port_table {
        key = {
            ig_md.min_id : exact;
        }
        actions = {
            set_deflect_port;
            drop;
        }
        size = TABLE_SIZE;
        default_action = drop();
    }

    action set_bucket_local_queue_occ(bit<5> bucket) {
        ig_md.bucket_local_queue_occ = bucket;
    }

    table encode_local_queue_occ {
        key     = { ig_md.local_queue_occ : range; }
        actions = { set_bucket_local_queue_occ; }
        size    = 17;
    }

    action set_bucket_min_cand_occ(bit<5> bucket) {
        ig_md.bucket_min_cand_occ = bucket;
    }

    table encode_min_cand_occ {
        key     = { ig_md.min_cand_occ : range; }
        actions = { set_bucket_min_cand_occ; }
        size    = 27;
    }

    action set_bucket_cand2_occ(bit<5> bucket) {
        ig_md.bucket_cand2_occ = bucket;
    }

    table encode_cand2_occ {
        key     = { ig_md.cand2_occ : range; }
        actions = { set_bucket_cand2_occ; }
        size    = 29;
    }

    action set_bucket_cand_occ_spread(bit<7> bucket) {
        ig_md.bucket_cand_occ_spread = bucket;
    }

    table encode_cand_occ_spread {
        key     = { ig_md.cand_occ_spread : range; }
        actions = { set_bucket_cand_occ_spread; }
        size    = 99;
    }

    action set_bucket_occ_vs_min_cand(bit<5> bucket) {
        ig_md.bucket_occ_vs_min_cand = bucket;
    }

    table encode_occ_vs_min_cand {
        key     = { ig_md.occ_vs : range; }
        actions = { set_bucket_occ_vs_min_cand; }
        size    = 25;
    }

    action set_bucket_occ_vs_mean_cand(bit<6> bucket) {
        ig_md.bucket_occ_vs_mean_cand = bucket;
    }

    table encode_occ_vs_mean_cand {
        key     = { ig_md.occ_vs : range; }
        actions = { set_bucket_occ_vs_mean_cand; }
        size    = 60;
    }

    action do_deflect() {
        ig_md.deflect = 1;
    }

    table model_decision {
        key = {
            ig_md.bucket_local_queue_occ : range;
            ig_md.bucket_min_cand_occ : range;
            ig_md.bucket_cand2_occ : range;
            ig_md.bucket_cand_occ_spread : range;
            ig_md.bucket_occ_vs_min_cand : range;
            ig_md.bucket_occ_vs_mean_cand : range;
        }
        actions     = { do_deflect; NoAction; }
        default_action = NoAction();
        size        = 32;
    }

    // Mux: select the local-port occupancy based on port_idx set by ipv4_lpm.
    action set_local_occ_q0() { ig_md.local_queue_occ = ig_md.q0_occ; }
    action set_local_occ_q1() { ig_md.local_queue_occ = ig_md.q1_occ; }
    action set_local_occ_q2() { ig_md.local_queue_occ = ig_md.q2_occ; }
    action set_local_occ_q3() { ig_md.local_queue_occ = ig_md.q3_occ; }
    table select_local_occ_tbl {
        key = { ig_md.port_idx : exact; }
        actions = { set_local_occ_q0; set_local_occ_q1;
                    set_local_occ_q2; set_local_occ_q3; nop; }
        const entries = {
            2w0 : set_local_occ_q0();
            2w1 : set_local_occ_q1();
            2w2 : set_local_occ_q2();
            2w3 : set_local_occ_q3();
        }
        default_action = nop();
        size = 4;
    }

    // Sorting-network level 1: pairwise comparisons + partial mean-sums.
    // Slot-reuse (overwritten in level2/level3):
    //   min_cand_occ ← min(q0,q1)   max1 ← max(q0,q1)
    //   cand2_occ    ← min(q2,q3)   cand_occ_spread ← max(q2,q3)
    //   sum          ← (q0>>2)+(q1>>2)   lo_hi ← (q2>>2)+(q3>>2)
    // All 6 outputs independent → single stage.
    action compute_level1() {
        ig_md.min_cand_occ    = min(ig_md.q0_occ, ig_md.q1_occ);
        ig_md.max1            = max(ig_md.q0_occ, ig_md.q1_occ);
        ig_md.cand2_occ       = min(ig_md.q2_occ, ig_md.q3_occ);
        ig_md.cand_occ_spread = max(ig_md.q2_occ, ig_md.q3_occ);
        ig_md.sum             = ig_md.q0_occ + ig_md.q1_occ;
        ig_md.lo_hi           = ig_md.q2_occ + ig_md.q3_occ;
        ig_md.occ_vs          = ig_md.q0_occ |-| ig_md.q1_occ;  // scratch: >0 iff q0 > q1
    }
    table level1_tbl {
        actions        = { compute_level1; }
        default_action = compute_level1();
        size           = 1;
    }

    // Sorting-network level 2: merge pairs into final min/max + intermediates.
    // sum computed first so it reads OLD lo_hi (= q2+q3 partial sum from level1).
    // All 5 outputs read DIFFERENT fields than any prior write → no intra-action deps.
    action compute_level2() {
        ig_md.sum          = ig_md.sum + ig_md.lo_hi;                  // (q0+q1)+(q2+q3) = sum*4
        ig_md.lo_hi        = max(ig_md.min_cand_occ, ig_md.cand2_occ); // max(min01,min23)
        ig_md.hi_lo        = min(ig_md.max1, ig_md.cand_occ_spread);   // min(max01,max23)
        ig_md.min_cand_occ = min(ig_md.min_cand_occ, ig_md.cand2_occ); // final overall min
        ig_md.max1         = max(ig_md.max1, ig_md.cand_occ_spread);   // final overall max
    }
    table level2_tbl {
        actions        = { compute_level2; }
        default_action = compute_level2();
        size           = 1;
    }

    apply {
        if(hdr.bee.isValid()){
            ig_intr_tm_md.ucast_egress_port[8:7] = ig_intr_md.ingress_port[8:7];
            ig_intr_tm_md.ucast_egress_port[6:0] = 68;
            if(hdr.bee.port_id == 0){
                write_ig_queue0_len_tbl.apply();
            } else if(hdr.bee.port_id == 1){
                write_ig_queue1_len_tbl.apply();
            } else if(hdr.bee.port_id == 2){
                write_ig_queue2_len_tbl.apply();
            } else if(hdr.bee.port_id == 3){
                write_ig_queue3_len_tbl.apply();
            }
        } else if(hdr.ipv4.isValid()){
            ipv4_lpm.apply();

            // All 4 reads are independent (no shared tmp) → compiler can parallelise.
            read_ig_queue0_len_tbl.apply();
            read_ig_queue1_len_tbl.apply();
            read_ig_queue2_len_tbl.apply();
            read_ig_queue3_len_tbl.apply();

            // Level-1: pairwise comparisons + partial sums + min_id scratch (one stage).
            select_local_occ_tbl.apply();
            level1_tbl.apply();

            // min_id: occ_vs = q0 |-| q1 from level1; >0 means q1 < q0 → port 1 is less loaded.
            if (ig_md.occ_vs != 0) { ig_md.min_id = 1; }

            // Level-2: merge pairs (single stage, no intra-action deps).
            level2_tbl.apply();

            // Level-3: second smallest = min of the two "middle" sorted elements.
            ig_md.cand2_occ = min(ig_md.lo_hi, ig_md.hi_lo);

            // Shift sum to get mean (single op → own auto-gen stage).
            ig_md.sum = ig_md.sum >> 2;
            // occ_vs_mean = local |-| mean.
            ig_md.occ_vs = ig_md.local_queue_occ |-| ig_md.sum;
            encode_occ_vs_mean_cand.apply();

            ig_md.cand_occ_spread = ig_md.max1 - ig_md.min_cand_occ;

            encode_cand2_occ.apply();
            encode_cand_occ_spread.apply();
            encode_min_cand_occ.apply();
            encode_local_queue_occ.apply();

            ig_md.occ_vs = ig_md.local_queue_occ |-| ig_md.min_cand_occ;
            encode_occ_vs_min_cand.apply();

            model_decision.apply();

            if (ig_md.deflect == 1) {
                deflect_port_table.apply();
            }
        }
    }
}

/*
    Switch Egress Pipeline
*/
control SwitchEgress(
        inout header_t hdr,
        inout metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprsr,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport) {

    action nop() { }

    action set_port_idx_in_reg(bit<2> logical_port) {
        eg_md.port_idx = logical_port;
    }


    #define DECLARE_EG_QUEUE_LEN_REG(N)                                        \
        Register<bit<16>, bit<1>>(1) eg_queue##N##_len_reg;                   \
        RegisterAction<bit<16>, bit<1>, bit<16>>(eg_queue##N##_len_reg)        \
        r_eg_queue##N##_len_alu = {                                            \
            void apply(inout bit<16> register_data, out bit<16> result) {      \
                result = register_data;                                         \
            }                                                                   \
        };                                                                     \
        RegisterAction<bit<16>, bit<1>, void>(eg_queue##N##_len_reg)           \
        w_eg_queue##N##_len_alu = {                                            \
            void apply(inout bit<16> register_data) {                          \
                register_data = (bit<16>)(eg_md.local_queue_occ);             \
            }                                                                   \
        };

    //TODO: da decidere logica, shiftare? Cappare la dim massima della coda?

    DECLARE_EG_QUEUE_LEN_REG(0)
    DECLARE_EG_QUEUE_LEN_REG(1)
    DECLARE_EG_QUEUE_LEN_REG(2)
    DECLARE_EG_QUEUE_LEN_REG(3)

    table set_port_idx_table {
        key = {
            eg_intr_md.egress_port : exact;

        }
        actions = {
            set_port_idx_in_reg;
            nop;
        }
        size = TABLE_SIZE;
        default_action = nop();
    }

    apply {
        if(hdr.bee.isValid()) {
            // Bee packet: load current egress queue depth into header so
            // ingress can update its register on the next loop iteration.
            if(hdr.bee.port_id == 0){
                hdr.bee.queue_occ = r_eg_queue0_len_alu.execute(0);
            } else if(hdr.bee.port_id == 1){
                hdr.bee.queue_occ = r_eg_queue1_len_alu.execute(0);
            } else if(hdr.bee.port_id == 2){
                hdr.bee.queue_occ = r_eg_queue2_len_alu.execute(0);
            } else if(hdr.bee.port_id == 3){
                hdr.bee.queue_occ = r_eg_queue3_len_alu.execute(0);
            }
        } else {
            // Data packet: snapshot egress queue depth into registers.
            set_port_idx_table.apply();
            eg_md.local_queue_occ = (bit<16>) eg_intr_md.deq_qdepth << 5;
            if(eg_intr_md.egress_port == 0){
                w_eg_queue0_len_alu.execute(0);
            } else if(eg_intr_md.egress_port == 1){
                w_eg_queue1_len_alu.execute(0);
            } else if(eg_intr_md.egress_port == 2){
                w_eg_queue2_len_alu.execute(0);
            } else if(eg_intr_md.egress_port == 3){
                w_eg_queue3_len_alu.execute(0);
            }
        }
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;
