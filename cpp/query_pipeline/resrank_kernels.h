// resrank_kernels.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// ============================================================
// Small AVX helpers
// ============================================================

static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float hsum256_ps_2(__m256 v0, __m256 v1) {
    // sum(v0) + sum(v1)
    return hsum256_ps(_mm256_add_ps(v0, v1));
}

// ============================================================
// Top10 fixed-k helpers (FAST: cached worst + early-reject)
//   - IP: keep TOP-10 (largest scores)
//   - L2: keep TOP-10 (smallest distances)
// ============================================================

struct Top10MaxFast {
    float   s[10];
    int64_t id[10];
    int     count;
    int     worst_i;
    float   worst_v;

    inline void reset() {
        count = 0;
        worst_i = 0;
        worst_v = -1e30f;
        for (int i = 0; i < 10; i++) {
            s[i] = -1e30f;
            id[i] = -1;
        }
    }

    inline void recompute_worst() {
        int wi = 0;
        float wv = s[0];
        for (int i = 1; i < 10; i++) {
            if (s[i] < wv) { wv = s[i]; wi = i; }
        }
        worst_i = wi;
        worst_v = wv;
    }

    inline void push(float score, int64_t idx) {
        if (count < 10) {
            s[count] = score;
            id[count] = idx;
            count++;
            if (count == 10) recompute_worst();
            return;
        }
        // early reject
        if (score <= worst_v) return;

        s[worst_i] = score;
        id[worst_i] = idx;
        recompute_worst();
    }
};

struct Top10MinFast {
    float   d[10];
    int64_t id[10];
    int     count;
    int     worst_i;   // "worst" for min-topk = largest distance
    float   worst_v;

    inline void reset() {
        count = 0;
        worst_i = 0;
        worst_v = 1e30f;
        for (int i = 0; i < 10; i++) {
            d[i] = 1e30f;
            id[i] = -1;
        }
    }

    inline void recompute_worst() {
        int wi = 0;
        float wv = d[0];
        for (int i = 1; i < 10; i++) {
            if (d[i] > wv) { wv = d[i]; wi = i; }
        }
        worst_i = wi;
        worst_v = wv;
    }

    inline void push(float dist, int64_t idx) {
        if (count < 10) {
            d[count] = dist;
            id[count] = idx;
            count++;
            if (count == 10) recompute_worst();
            return;
        }
        // early reject
        if (dist >= worst_v) return;

        d[worst_i] = dist;
        id[worst_i] = idx;
        recompute_worst();
    }
};

// Array-friendly resets (for your bench local buffers)
static inline void top10_reset_max(float* S10, int64_t* I10) {
    for (int i = 0; i < 10; i++) { S10[i] = -1e30f; I10[i] = -1; }
}
static inline void top10_reset_min(float* D10, int64_t* I10) {
    for (int i = 0; i < 10; i++) { D10[i] = 1e30f; I10[i] = -1; }
}

// Merge helper: push 10 pairs from (src) into (dst) using FAST struct
static inline void top10_merge_max(float* dstS, int64_t* dstI,
                                  const float* srcS, const int64_t* srcI) {
    Top10MaxFast tk;
    // initialize from dst
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (dstI[i] >= 0) tk.push(dstS[i], dstI[i]);
    }
    // push from src
    for (int i = 0; i < 10; i++) {
        if (srcI[i] >= 0) tk.push(srcS[i], srcI[i]);
    }
    // write back
    for (int i = 0; i < 10; i++) {
        dstS[i] = tk.s[i];
        dstI[i] = tk.id[i];
    }
}

static inline void top10_merge_min(float* dstD, int64_t* dstI,
                                  const float* srcD, const int64_t* srcI) {
    Top10MinFast tk;
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (dstI[i] >= 0) tk.push(dstD[i], dstI[i]);
    }
    for (int i = 0; i < 10; i++) {
        if (srcI[i] >= 0) tk.push(srcD[i], srcI[i]);
    }
    for (int i = 0; i < 10; i++) {
        dstD[i] = tk.d[i];
        dstI[i] = tk.id[i];
    }
}

// ============================================================
// (1) uint8 / d=128 / exact L2^2 (AVX2)
// ============================================================
// Return L2^2(q,x) where q,x are uint8[128]
static inline float l2_u8_d128_avx2(const uint8_t* q, const uint8_t* x) {
    __m256i acc = _mm256_setzero_si256();

    // 128 bytes -> 4 blocks of 32 bytes
    for (int off = 0; off < 128; off += 32) {
        __m128i q0_128 = _mm_loadu_si128((const __m128i*)(q + off));
        __m128i x0_128 = _mm_loadu_si128((const __m128i*)(x + off));
        __m128i q1_128 = _mm_loadu_si128((const __m128i*)(q + off + 16));
        __m128i x1_128 = _mm_loadu_si128((const __m128i*)(x + off + 16));

        __m256i q0 = _mm256_cvtepu8_epi16(q0_128);
        __m256i x0 = _mm256_cvtepu8_epi16(x0_128);
        __m256i q1 = _mm256_cvtepu8_epi16(q1_128);
        __m256i x1 = _mm256_cvtepu8_epi16(x1_128);

        __m256i d0 = _mm256_sub_epi16(q0, x0);
        __m256i d1 = _mm256_sub_epi16(q1, x1);

        __m256i s0 = _mm256_madd_epi16(d0, d0);
        __m256i s1 = _mm256_madd_epi16(d1, d1);

        acc = _mm256_add_epi32(acc, s0);
        acc = _mm256_add_epi32(acc, s1);
    }

    alignas(32) int32_t tmp[8];
    _mm256_store_si256((__m256i*)tmp, acc);
    int32_t sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return (float)sum;
}

// ============================================================
// (2) float32 / d=96 / exact L2^2 (AVX2)
// ============================================================

static inline float l2_f32_d96_avx2(const float* __restrict a,
                                   const float* __restrict b) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    // 96 = 12 * 8
    for (int off = 0; off < 96; off += 16) {
        __m256 va0 = _mm256_loadu_ps(a + off);
        __m256 vb0 = _mm256_loadu_ps(b + off);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
#endif

        __m256 va1 = _mm256_loadu_ps(a + off + 8);
        __m256 vb1 = _mm256_loadu_ps(b + off + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
#if defined(__FMA__)
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
#else
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
#endif
    }
    return hsum256_ps_2(acc0, acc1);
}

// ============================================================
// (3) float32 / d=1536 / exact IP (AVX2 + FMA)
// ============================================================

static inline float ip_f32_d1536_avx2(const float* __restrict a,
                                     const float* __restrict b) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    // 1536 = 192 * 8
    // Unroll by 16 floats per iter (2 * 8)
    for (int off = 0; off < 1536; off += 16) {
        __m256 va0 = _mm256_loadu_ps(a + off);
        __m256 vb0 = _mm256_loadu_ps(b + off);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
#endif

        __m256 va1 = _mm256_loadu_ps(a + off + 8);
        __m256 vb1 = _mm256_loadu_ps(b + off + 8);
#if defined(__FMA__)
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
#else
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
#endif
    }
    return hsum256_ps_2(acc0, acc1);
}

// ============================================================
// (3b) float32 / d=1536 / exact L2^2 (AVX2 + FMA)
// ============================================================

static inline float l2_f32_d1536_avx2(const float* __restrict a,
                                     const float* __restrict b) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    // 1536 = 192 * 8
    // Unroll by 16 floats per iter (2 * 8)
    for (int off = 0; off < 1536; off += 16) {
        __m256 va0 = _mm256_loadu_ps(a + off);
        __m256 vb0 = _mm256_loadu_ps(b + off);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
#endif

        __m256 va1 = _mm256_loadu_ps(a + off + 8);
        __m256 vb1 = _mm256_loadu_ps(b + off + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
#if defined(__FMA__)
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
#else
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
#endif
    }
    return hsum256_ps_2(acc0, acc1);
}

// ============================================================
// Scan APIs (NO ids_buf, use id_base + j)
//   - D10 / I10 are updated in-place (caller decides reset)
// ============================================================

// ---- uint8 d=128 L2^2 ----
void scan_u8_d128_l2_top10_oneq_idbase(
    const uint8_t* q_u8,            // [128]
    const uint8_t* xb_u8,           // [nb,128]
    int64_t id_base,                // global_id_base
    size_t nb,
    float* D10,                     // [10] keep MIN
    int64_t* I10
);

// ---- float32 d=96 L2^2 ----
void scan_f32_d96_l2_top10_oneq_idbase(
    const float* q,                // [96]
    const float* xb,               // [nb,96]
    int64_t id_base,
    size_t nb,
    float* D10,                    // [10] keep MIN
    int64_t* I10
);

// ---- float32 d=1536 IP ----
void scan_f32_d1536_ip_top10_oneq_idbase(
    const float* q,                // [1536]
    const float* xb,               // [nb,1536]
    int64_t id_base,
    size_t nb,
    float* S10,                    // [10] keep MAX
    int64_t* I10
);

// ---- float32 d=1536 L2^2 ----
void scan_f32_d1536_l2_top10_oneq_idbase(
    const float* q,                // [1536]
    const float* xb,               // [nb,1536]
    int64_t id_base,
    size_t nb,
    float* D10,                    // [10] keep MIN
    int64_t* I10
);



// ============================================================
// Low-dim float32 L2^2 (D <= 32) + cluster-centroid selection
// ============================================================

// Compute ||q-x||^2 for float32 vectors, D in [1,32]. Uses AVX2 when possible.
inline float l2_f32_dle32_avx2(const float* q, const float* x, int D) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= D; i += 8) {
        __m256 qv = _mm256_loadu_ps(q + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 dv = _mm256_sub_ps(qv, xv);
        acc = _mm256_fmadd_ps(dv, dv, acc);
    }
    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out = _mm_cvtss_f32(sum);
    // tail
    for (; i < D; i++) {
        float d = q[i] - x[i];
        out += d * d;
    }
    return out;
}

// Batch centroid distances: out_dist[k] = ||q - centroids[k]||^2, centroids is row-major [K,D].
void l2_f32_centroids_dle32_l2sq(
    const float* q,                 // [D]
    const float* centroids,         // [K,D]
    int D,
    int K,
    float* out_dist                 // [K]
);

// Select clusters by centroid distance.
// Returns number selected (<= sel_cap).
// Outputs:
//   sel_idx[i] : selected cluster index in [0,K)
//   sel_d[i]   : corresponding centroid L2^2 distance
//   *best_d    : global best (min over all K clusters)
int select_clusters_topm_delta_ymax_l2sq(
    const float* dist_l2sq,         // [K]
    int K,
    int topM,                       // <=0 disables topM
    float delta_l2sq,               // <0 disables delta band (best + delta)
//  float ymax_l2sq : if >=0, also require dist <= ymax_l2sq (typically from your y_max fn)
    float ymax_l2sq,                // <0 disables
    int* sel_idx,                   // [sel_cap]
    float* sel_d,                   // [sel_cap]
    int sel_cap,
    float* best_d                   // out
);

// Scan float32 codes with D<=32, ids provided per row. Keeps MIN top10.
void scan_f32_dle32_l2_top10_oneq_ids(
    const float* q,                 // [D]
    const float* xb,                // [nb,D] row-major
    int D,
    const int64_t* ids,             // [nb]
    size_t nb,
    float* D10,                     // [10]
    int64_t* I10                    // [10]
);
