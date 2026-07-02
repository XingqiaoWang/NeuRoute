// resrank_kernels.cpp
#include "resrank_kernels.h"
#include <cmath>

void scan_u8_d128_l2_top10_oneq_idbase(
    const uint8_t* q_u8,
    const uint8_t* xb_u8,
    int64_t id_base,
    size_t nb,
    float* D10,
    int64_t* I10
) {
    // Load current top10 into a fast struct (supports early-reject)
    Top10MinFast tk;
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (I10[i] >= 0) tk.push(D10[i], I10[i]);
    }

    for (size_t j = 0; j < nb; j++) {
        const uint8_t* x = xb_u8 + j * 128;
        float dist = l2_u8_d128_avx2(q_u8, x);
        tk.push(dist, id_base + (int64_t)j);
    }

    // Write back
    for (int i = 0; i < 10; i++) {
        D10[i] = tk.d[i];
        I10[i] = tk.id[i];
    }
}

void scan_f32_d96_l2_top10_oneq_idbase(
    const float* q,
    const float* xb,
    int64_t id_base,
    size_t nb,
    float* D10,
    int64_t* I10
) {
    Top10MinFast tk;
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (I10[i] >= 0) tk.push(D10[i], I10[i]);
    }

    for (size_t j = 0; j < nb; j++) {
        const float* x = xb + j * 96;
        float dist = l2_f32_d96_avx2(q, x);
        tk.push(dist, id_base + (int64_t)j);
    }

    for (int i = 0; i < 10; i++) {
        D10[i] = tk.d[i];
        I10[i] = tk.id[i];
    }
}

void scan_f32_d1536_ip_top10_oneq_idbase(
    const float* q,
    const float* xb,
    int64_t id_base,
    size_t nb,
    float* S10,
    int64_t* I10
) {
    Top10MaxFast tk;
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (I10[i] >= 0) tk.push(S10[i], I10[i]);
    }

    for (size_t j = 0; j < nb; j++) {
        const float* x = xb + j * 1536;
        float score = ip_f32_d1536_avx2(q, x);
        tk.push(score, id_base + (int64_t)j);
    }

    for (int i = 0; i < 10; i++) {
        S10[i] = tk.s[i];
        I10[i] = tk.id[i];
    }
}

void scan_f32_d1536_l2_top10_oneq_idbase(
    const float* q,
    const float* xb,
    int64_t id_base,
    size_t nb,
    float* D10,
    int64_t* I10
) {
    Top10MinFast tk;
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (I10[i] >= 0) tk.push(D10[i], I10[i]);
    }

    for (size_t j = 0; j < nb; j++) {
        const float* x = xb + j * 1536;
        float dist = l2_f32_d1536_avx2(q, x);
        tk.push(dist, id_base + (int64_t)j);
    }

    for (int i = 0; i < 10; i++) {
        D10[i] = tk.d[i];
        I10[i] = tk.id[i];
    }
}



void l2_f32_centroids_dle32_l2sq(
    const float* q,
    const float* centroids,
    int D,
    int K,
    float* out_dist
) {
    for (int k = 0; k < K; k++) {
        out_dist[k] = l2_f32_dle32_avx2(q, centroids + (size_t)k * (size_t)D, D);
    }
}

int select_clusters_topm_delta_ymax_l2sq(
    const float* dist_l2sq,
    int K,
    int topM,
    float delta_l2sq,
    float ymax_l2sq,
    int* sel_idx,
    float* sel_d,
    int sel_cap,
    float* best_d
) {
    if (K <= 0 || sel_cap <= 0) { if (best_d) *best_d = 0.0f; return 0; }

    // best (min)
    float best = dist_l2sq[0];
    int besti = 0;
    for (int k = 1; k < K; k++) {
        float d = dist_l2sq[k];
        if (d < best) { best = d; besti = k; }
    }
    if (best_d) *best_d = best;

    float thr = INFINITY;
    if (delta_l2sq >= 0.0f) thr = best + delta_l2sq;
    if (ymax_l2sq  >= 0.0f) thr = (thr < ymax_l2sq ? thr : ymax_l2sq);

    int out_n = 0;

    // 1) TopM selection (by smallest dist), optionally respecting thr if enabled.
    if (topM > 0) {
        // insertion into small arrays (O(K*topM)) - good for small topM.
        const int M = (topM < sel_cap ? topM : sel_cap);
        // initialize with +inf
        for (int i = 0; i < M; i++) { sel_d[i] = INFINITY; sel_idx[i] = -1; }
        for (int k = 0; k < K; k++) {
            float d = dist_l2sq[k];
            if (!(d < sel_d[M-1])) continue;
            if (d > thr) continue; // if thr active, enforce
            // insert
            int pos = M - 1;
            while (pos > 0 && d < sel_d[pos-1]) {
                sel_d[pos] = sel_d[pos-1];
                sel_idx[pos] = sel_idx[pos-1];
                pos--;
            }
            sel_d[pos] = d;
            sel_idx[pos] = k;
        }
        // compact valid
        out_n = 0;
        for (int i = 0; i < M; i++) {
            if (sel_idx[i] >= 0) {
                sel_idx[out_n] = sel_idx[i];
                sel_d[out_n]   = sel_d[i];
                out_n++;
            }
        }
    }

    // 2) Delta/ymax band union: add any k with dist<=thr (or if thr inactive, skip).
    if (thr < INFINITY) {
        for (int k = 0; k < K && out_n < sel_cap; k++) {
            float d = dist_l2sq[k];
            if (d > thr) continue;
            // already selected?
            bool exists = false;
            for (int i = 0; i < out_n; i++) {
                if (sel_idx[i] == k) { exists = true; break; }
            }
            if (!exists) {
                sel_idx[out_n] = k;
                sel_d[out_n]   = d;
                out_n++;
            }
        }
    } else {
        // If no thr and no topM, at least return best
        if (topM <= 0) {
            sel_idx[0] = besti;
            sel_d[0] = best;
            out_n = 1;
        }
    }

    return out_n;
}

void scan_f32_dle32_l2_top10_oneq_ids(
    const float* q,
    const float* xb,
    int D,
    const int64_t* ids,
    size_t nb,
    float* D10,
    int64_t* I10
) {
    Top10MinFast tk;
    tk.reset();
    for (int i = 0; i < 10; i++) {
        if (I10[i] >= 0) tk.push(D10[i], I10[i]);
    }

    for (size_t j = 0; j < nb; j++) {
        const float* x = xb + j * (size_t)D;
        float dist = l2_f32_dle32_avx2(q, x, D);
        tk.push(dist, ids[j]);
    }

    for (int i = 0; i < 10; i++) {
        D10[i] = tk.d[i];
        I10[i] = tk.id[i];
    }
}
