#ifndef _D8_HEADERS_
#define _D8_HEADERS_

#include "defines.p4"

// Ethernet Header
header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

// IPv4 Header
header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

// Bee Header (Worker Packet)
header bee_t {
    bit<8> port_id;
    bit<16> queue_occ;
}

struct header_t {
    ethernet_t ethernet;
    ipv4_t     ipv4;
    bee_t      bee;
}

// User Metadata
struct metadata_t {
    // Forwarding
    bit<2> port_idx;
    bit<2> min_id;
    bit<1>  deflect;

    bit<16> local_queue_occ;
    bit<16> min_cand_occ;
    bit<16> max1;
    bit<16> cand2_occ;
    bit<16> cand_occ_spread;
    bit<16> occ_vs;
    bit<16> q0_occ;
    bit<16> q1_occ;
    bit<16> q2_occ;
    bit<16> q3_occ;
    bit<16> sum;
    bit<16> lo_hi;
    bit<16> hi_lo;

    // Bucketed features for model_decision table
    bit<6>  bucket_local_queue_occ;
    bit<6>  bucket_min_cand_occ;
    bit<6>  bucket_cand2_occ;
    bit<8>  bucket_cand_occ_spread;
    bit<7>  bucket_occ_vs_min_cand;
    bit<7>  bucket_occ_vs_mean_cand;
}

#endif
