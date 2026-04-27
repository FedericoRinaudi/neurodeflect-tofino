#ifndef _D1_TREE_DEFINES_
#define _D1_TREE_DEFINES_

// Constants
#define RECIRCULATE_PORT 68
#define CPU_PORT 64
#define DROP_PORT 511

#define ETHERTYPE_IPV4 0x0800
#define ETHERTYPE_BEE 0x0801
#define ETHERTYPE_MIN_BEE 0x0802

#define IP_PROTOCOLS_TCP 6
#define IP_PROTOCOLS_UDP 17

#define TABLE_SIZE 512
#define INGRESS_CTR_SIZE 65536
#define CTR_WIDTH 64
#define INGRESS_CTR_WIDTH 16

#define NUM_QUEUES 4

// Threshold for deflection (Can be tuned)
#define THRESHOLD_OCC 1000
#define THRESHOLD_OCC2 1200
#define THRESHOLD_OCC3 1100
#define THRESHOLD_OCC_IDX2 1000
#define THRESHOLD_TOT_OCC 3000
#define THRESHOLD_TOT_OCC_IDX0 4000
#define THRESHOLD_TOT_OCC_IDX1 4000
#define THRESHOLD_OCC_IDX2_2 3500
#define THRESHOLD_OCC_IDX2_3 3600

// Widths
// typedef bit<9>  PortId_t; // Defined in tna.p4
// typedef bit<16> QueueId_t; // Defined in tna.p4

#endif
