#ifndef _D8_PARSER_
#define _D8_PARSER_

#include "headers.p4"

parser SwitchIngressParser(packet_in pkt,
    out header_t hdr,
    out metadata_t ig_md,
    out ingress_intrinsic_metadata_t ig_intr_md)
{
    state start {
        pkt.extract(ig_intr_md);

        ig_md.port_idx               = 0;
        ig_md.min_id                 = 0;
        ig_md.deflect                = 0;
        ig_md.local_queue_occ        = 0;
        ig_md.min_cand_occ           = 0;
        ig_md.max1                   = 0;
        ig_md.cand2_occ              = 0;
        ig_md.cand_occ_spread        = 0;
        ig_md.occ_vs                 = 0;
        ig_md.bucket_local_queue_occ = 0;
        ig_md.bucket_min_cand_occ    = 0;
        ig_md.bucket_cand2_occ       = 0;
        ig_md.bucket_cand_occ_spread = 0;
        ig_md.bucket_occ_vs_min_cand = 0;
        ig_md.bucket_occ_vs_mean_cand = 0;

        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            ETHERTYPE_IPV4 : parse_ipv4;
            ETHERTYPE_BEE  : parse_bee;
            default        : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

    state parse_bee {
        pkt.extract(hdr.bee);
        transition accept;
    }
}

control SwitchIngressDeparser(packet_out pkt,
    inout header_t hdr,
    in metadata_t ig_md,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    in ingress_intrinsic_metadata_t ig_intr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

parser SwitchEgressParser(packet_in pkt,
    out header_t hdr,
    out metadata_t eg_md,
    out egress_intrinsic_metadata_t eg_intr_md)
{
    state start {
        pkt.extract(eg_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            ETHERTYPE_IPV4 : parse_ipv4;
            ETHERTYPE_BEE  : parse_bee;
            default        : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

    state parse_bee {
        pkt.extract(hdr.bee);
        transition accept;
    }
}

control SwitchEgressDeparser(packet_out pkt,
    inout header_t hdr,
    in metadata_t eg_md,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    in egress_intrinsic_metadata_t eg_intr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

#endif
