#pragma once
// query_bins_bins_only_v3a_standalone.hpp
//
// This header is a *standalone* upgrade of your original:
//   query_bins_bins_only_v3a.hpp  (FULL-DP bins-only generator)
//
// Changes vs original v3a:
//   1) No dependency on query_bins_pruned_v2.hpp. We embed the needed qbpg_v1
//      types + helpers (and keep the same names) so existing bench code can
//      include ONLY this file.
//   2) FULL-DP bins generator now honors filt.keep_base:
//        - keep_base=false (default): enumerate masks i=1..M-1 (same as before)
//        - keep_base=true: include i=0 (mask=0, score=0) so base_id is inserted
//          into bins (typically bin0)
//   3) We also embed the PRUNED generator + build_query_bins_opt from pruned_v2
//      for convenience, so you can switch between pruned and full-DP without
//      adding another include.
//
// Notes:
//   - FULL-DP path still outputs only NORMAL bins (CSR of bucket ids). BIG split
//     is typically handled in the bench after bins are produced (same as before).
//   - PRUNED path (qbpg_v1::enumerate_pruned + build_query_bins_opt) supports
//     optional BIG split internally.
//
// Standard library only.

#include <cstdint>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <utility>
#include <stdexcept>

// =====================================================================================
// qbpg_v1  (embedded from query_bins_pruned_v2.hpp)
// =====================================================================================
namespace qbpg_v1 {

struct BucketScore {
    uint32_t bid;
    float    score;
};

enum class BucketTag : uint8_t { NORMAL = 0, BIG = 1, SKIP = 2 };

struct QueryBins {
    int NBINS = 0;
    // ids[offsets[b] .. offsets[b+1]) are bucket ids for bin b.
    std::vector<uint32_t> ids;
    std::vector<uint32_t> offsets;   // size NBINS+1
    std::vector<uint64_t> vec_sum;   // optional per-bin sum of bucket sizes

    void init(int nbins) {
        NBINS = nbins;
        ids.clear();
        offsets.assign((size_t)NBINS + 1, 0u);
        vec_sum.assign((size_t)NBINS, 0ull);
    }
    void clear() {
        ids.clear();
        std::fill(offsets.begin(), offsets.end(), 0u);
        std::fill(vec_sum.begin(), vec_sum.end(), 0ull);
    }
};

struct EnumFilters {
    float max_score = std::numeric_limits<float>::infinity();
    bool  enable_budget = false;
    float budget = 0.0f;
    bool  keep_base = false;  // include base_id(score=0)
};

// logits/thr -> base_id + (cost_rank, rank2bit). D<=32.
inline void prepare_inputs_from_logits_thr_absdiff(
    const float* logits,
    const float* thr,
    int D,
    int Lsel,
    uint32_t& out_base_id,
    std::vector<float>& out_cost_rank,
    std::vector<uint32_t>& out_rank2bit
) {
    if (D <= 0) { out_base_id = 0u; out_cost_rank.clear(); out_rank2bit.clear(); return; }
    if (D > 32) throw std::runtime_error("prepare_inputs_from_logits_thr_absdiff: D>32 not supported");
    if (Lsel > D) Lsel = D;
    if (Lsel < 0) Lsel = 0;

    uint32_t base = 0u;
    for (int d = 0; d < D; ++d) {
        if (logits[d] >= thr[d]) base |= (1u << (uint32_t)d);
    }
    out_base_id = base;

    struct Pair { float c; int d; };
    std::vector<Pair> v;
    v.reserve((size_t)D);
    for (int d = 0; d < D; ++d) {
        float c = logits[d] - thr[d];
        if (c < 0) c = -c;
        v.push_back({c, d});
    }
    auto cmp = [](const Pair& a, const Pair& b) {
        if (a.c < b.c) return true;
        if (a.c > b.c) return false;
        return a.d < b.d;
    };

    if (Lsel == 0) { out_cost_rank.clear(); out_rank2bit.clear(); return; }

    std::nth_element(v.begin(), v.begin() + Lsel, v.end(), cmp);
    v.resize((size_t)Lsel);
    std::sort(v.begin(), v.end(), cmp);

    out_cost_rank.resize((size_t)Lsel);
    out_rank2bit.resize((size_t)Lsel);
    for (int r = 0; r < Lsel; ++r) {
        int d = v[(size_t)r].d;
        out_cost_rank[(size_t)r] = v[(size_t)r].c;
        out_rank2bit[(size_t)r] = (1u << (uint32_t)d);
    }
}

// PRUNED enumeration (combination DFS with pruning).
inline void enumerate_pruned(
    uint32_t base_id,
    const float* cost_rank,
    const uint32_t* rank2bit,
    int Lsel,
    int Rmax,
    const EnumFilters& filt,
    std::vector<BucketScore>& out_items
) {
    out_items.clear();
    if (Lsel <= 0 || Rmax < 0) return;
    if (Rmax > Lsel) Rmax = Lsel;

    float limit = filt.max_score;
    if (filt.enable_budget && filt.budget < limit) limit = filt.budget;

    if (filt.keep_base) {
        if (0.0f <= limit) out_items.push_back({base_id, 0.0f});
    }
    if (!(limit >= 0.0f)) return;

    struct Node {
        int16_t depth;
        int16_t next_r;
        float   sum;
        uint32_t mask;
    };

    std::vector<Node> st;
    st.reserve((size_t)Rmax * 128);

    for (int r = 0; r < Lsel; ++r) {
        float s = cost_rank[r];
        if (s > limit) break;
        uint32_t m = rank2bit[r];
        out_items.push_back({base_id ^ m, s});
        if (Rmax >= 2) st.push_back(Node{1, (int16_t)(r + 1), s, m});
    }

    while (!st.empty()) {
        Node cur = st.back();
        st.pop_back();
        if (cur.depth >= Rmax) continue;

        for (int r = cur.next_r; r < Lsel; ++r) {
            float s2 = cur.sum + cost_rank[r];
            if (s2 > limit) break;
            uint32_t m2 = cur.mask ^ rank2bit[r];
            out_items.push_back({base_id ^ m2, s2});
            int16_t depth2 = (int16_t)(cur.depth + 1);
            if (depth2 < Rmax) st.push_back(Node{depth2, (int16_t)(r + 1), s2, m2});
        }
    }
}

// Bin + optional BIG extraction.
template <class ScoreToBinFn, class ClassifyFn, class BucketSizeFn>
inline void build_query_bins_opt(
    const std::vector<BucketScore>& items,
    int NBINS,
    ScoreToBinFn score_to_bin,
    ClassifyFn classify_bucket,
    BucketSizeFn bucket_size_of,
    bool enable_big_split,
    QueryBins& out_qb,
    std::vector<BucketScore>& out_big
) {
    out_qb.init(NBINS);
    out_big.clear();

    std::vector<uint32_t> cnt((size_t)NBINS, 0u);

    for (const auto& it : items) {
        BucketTag tag = classify_bucket(it.bid, it.score);
        if (tag == BucketTag::SKIP) continue;
        if (!enable_big_split) tag = BucketTag::NORMAL;
        if (tag == BucketTag::BIG) { out_big.push_back(it); continue; }

        int b = score_to_bin(it.score);
        if (b < 0) b = 0;
        if (b >= NBINS) b = NBINS - 1;
        cnt[(size_t)b] += 1u;
        out_qb.vec_sum[(size_t)b] += (uint64_t)bucket_size_of(it.bid);
    }

    out_qb.offsets[0] = 0u;
    for (int b = 0; b < NBINS; ++b) {
        out_qb.offsets[(size_t)b + 1] = out_qb.offsets[(size_t)b] + cnt[(size_t)b];
    }
    out_qb.ids.resize((size_t)out_qb.offsets[(size_t)NBINS]);

    std::vector<uint32_t> cur = out_qb.offsets;
    for (const auto& it : items) {
        BucketTag tag = classify_bucket(it.bid, it.score);
        if (tag == BucketTag::SKIP) continue;
        if (!enable_big_split) tag = BucketTag::NORMAL;
        if (tag != BucketTag::NORMAL) continue;

        int b = score_to_bin(it.score);
        if (b < 0) b = 0;
        if (b >= NBINS) b = NBINS - 1;
        uint32_t pos = cur[(size_t)b]++;
        out_qb.ids[(size_t)pos] = it.bid;
    }
}

} // namespace qbpg_v1

// =====================================================================================
// qbpg_bins_only_v3a  (FULL-DP bins-only generator)
// =====================================================================================
namespace qbpg_bins_only_v3a {

struct MaskPlan {
    int Lsel = 0;
    int Rmax = 0;
    std::vector<int32_t> parent;
    std::vector<int16_t> add_pos;
    int M() const { return (int)parent.size(); }
};

inline MaskPlan build_plan(int Lsel, int Rmax) {
    if (Lsel < 0) Lsel = 0;
    if (Rmax < 0) Rmax = 0;
    if (Rmax > Lsel) Rmax = Lsel;

    MaskPlan plan;
    plan.Lsel = Lsel;
    plan.Rmax = Rmax;
    plan.parent.reserve(80000);
    plan.add_pos.reserve(80000);
    plan.parent.push_back(-1);
    plan.add_pos.push_back(-1);

    std::vector<int32_t> idx_of((size_t)1u << (uint32_t)Lsel, -1);
    idx_of[0] = 0;

    std::vector<int> comb;
    comb.reserve((size_t)Rmax);

    std::function<void(int,int,uint32_t)> rec = [&](int start, int kleft, uint32_t mask) {
        if (kleft == 0) {
            int last_pos = comb.back();
            uint32_t parent_mask = mask ^ (1u << (uint32_t)last_pos);
            int32_t pidx = idx_of[(size_t)parent_mask];
            if (pidx < 0) std::abort();
            int32_t my = (int32_t)plan.parent.size();
            plan.parent.push_back(pidx);
            plan.add_pos.push_back((int16_t)last_pos);
            idx_of[(size_t)mask] = my;
            return;
        }
        for (int pos = start; pos <= Lsel - kleft; ++pos) {
            comb.push_back(pos);
            rec(pos + 1, kleft - 1, mask | (1u << (uint32_t)pos));
            comb.pop_back();
        }
    };

    for (int k = 1; k <= Rmax; ++k) rec(0, k, 0u);
    return plan;
}

struct BinsScratch {
    std::vector<uint32_t> cnt;
    std::vector<uint32_t> cur;
    void ensure(int NBINS) {
        if ((int)cnt.size() != NBINS) cnt.assign((size_t)NBINS, 0u);
        else std::fill(cnt.begin(), cnt.end(), 0u);
        if ((int)cur.size() != NBINS + 1) cur.resize((size_t)NBINS + 1);
    }
};

struct AlphaBinner {
    int NBINS = 64;
    float max_score = 0.3f;
    float alpha = 1.0f;
    inline int operator()(float score) const {
        if (NBINS <= 1) return 0;
        if (!(max_score > 0.0f)) return 0;
        float x = score / max_score;
        if (x < 0.0f) x = 0.0f;
        if (x > 1.0f) x = 1.0f;
        float y = (alpha == 1.0f) ? x : std::pow(x, alpha);
        int b = (int)(y * (float)NBINS);
        if (b < 0) b = 0;
        if (b >= NBINS) b = NBINS - 1;
        return b;
    }
};

inline void copy_cnt_prefix(uint32_t* out_cnt_prefix, int KOBS, const uint32_t* cnt_ptr, int NBINS) {
    if (!out_cnt_prefix || KOBS <= 0) return;
    int m = (KOBS < NBINS) ? KOBS : NBINS;
    if (!cnt_ptr) {
        for (int i = 0; i < KOBS; ++i) out_cnt_prefix[i] = 0u;
        return;
    }
    for (int b = 0; b < m; ++b) out_cnt_prefix[b] = cnt_ptr[b];
    for (int b = m; b < KOBS; ++b) out_cnt_prefix[b] = 0u;
}

// FULL-DP bins-only generation.
// Returns kept_raw: number of masks that passed (score<=limit), before cap.
//
// IMPORTANT upgrade:
//   If filt.keep_base==true, we include i=0 (mask=0, score=0) and thus base_id is
//   inserted into bins (typically bin0). This matches your "base bucket 正常加入bin".

template <class ScoreToBinFn>
inline int build_bins_full_dp(
    const MaskPlan& plan,
    uint32_t base_id,
    const float* cost_rank,
    const uint32_t* rank2bit,
    const qbpg_v1::EnumFilters& filt,
    int NBINS,
    ScoreToBinFn score_to_bin,
    int cap_per_bin,
    std::vector<float>& tmp_scores,
    std::vector<uint32_t>& tmp_maskdim,
    qbpg_v1::QueryBins& out_qb,
    BinsScratch* scratch = nullptr,
    uint32_t* out_cnt_prefix = nullptr,
    int KOBS = 0
) {
    const int M = plan.M();
    if (M <= 0) {
        out_qb.init(NBINS);
        copy_cnt_prefix(out_cnt_prefix, KOBS, nullptr, NBINS);
        return 0;
    }

    float limit = filt.max_score;
    if (filt.enable_budget && filt.budget < limit) limit = filt.budget;

    tmp_scores.resize((size_t)M);
    tmp_maskdim.resize((size_t)M);
    tmp_scores[0] = 0.0f;
    tmp_maskdim[0] = 0u;

    for (int i = 1; i < M; ++i) {
        int32_t p = plan.parent[(size_t)i];
        int16_t a = plan.add_pos[(size_t)i];
        tmp_scores[(size_t)i]  = tmp_scores[(size_t)p] + cost_rank[(int)a];
        tmp_maskdim[(size_t)i] = tmp_maskdim[(size_t)p] ^ rank2bit[(int)a];
    }

    const int i0 = filt.keep_base ? 0 : 1;

    if (NBINS <= 0) {
        int kept = 0;
        for (int i = i0; i < M; ++i) kept += (tmp_scores[(size_t)i] <= limit);
        return kept;
    }

    out_qb.init(NBINS);

    const size_t max_ids = (M > i0) ? (size_t)(M - i0) : 0;
    out_qb.ids.clear();
    if (out_qb.ids.capacity() < max_ids) out_qb.ids.reserve(max_ids);

    std::vector<uint32_t> local_cnt;
    std::vector<uint32_t> local_cur;
    uint32_t* cnt_ptr = nullptr;
    uint32_t* cur_ptr = nullptr;

    if (scratch) {
        scratch->ensure(NBINS);
        cnt_ptr = scratch->cnt.data();
        cur_ptr = scratch->cur.data();
    } else {
        local_cnt.assign((size_t)NBINS, 0u);
        local_cur.resize((size_t)NBINS + 1);
        cnt_ptr = local_cnt.data();
        cur_ptr = local_cur.data();
    }

    int kept_raw = 0;

    if (cap_per_bin <= 0) {
        for (int i = i0; i < M; ++i) {
            float s = tmp_scores[(size_t)i];
            if (s > limit) continue;
            kept_raw++;
            int b = score_to_bin(s);
            if (b < 0) b = 0;
            if (b >= NBINS) b = NBINS - 1;
            cnt_ptr[b] += 1u;
        }
    } else {
        for (int i = i0; i < M; ++i) {
            float s = tmp_scores[(size_t)i];
            if (s > limit) continue;
            kept_raw++;
            int b = score_to_bin(s);
            if (b < 0) b = 0;
            if (b >= NBINS) b = NBINS - 1;
            uint32_t c = cnt_ptr[b];
            if ((int)c < cap_per_bin) cnt_ptr[b] = c + 1u;
        }
    }

    copy_cnt_prefix(out_cnt_prefix, KOBS, cnt_ptr, NBINS);

    out_qb.offsets[0] = 0u;
    for (int b = 0; b < NBINS; ++b) {
        out_qb.offsets[(size_t)b + 1] = out_qb.offsets[(size_t)b] + cnt_ptr[b];
    }
    const uint32_t total = out_qb.offsets[(size_t)NBINS];

    out_qb.ids.resize((size_t)total);

    for (int b = 0; b <= NBINS; ++b) cur_ptr[b] = out_qb.offsets[(size_t)b];

    if (cap_per_bin <= 0) {
        for (int i = i0; i < M; ++i) {
            float s = tmp_scores[(size_t)i];
            if (s > limit) continue;
            int b = score_to_bin(s);
            if (b < 0) b = 0;
            if (b >= NBINS) b = NBINS - 1;
            uint32_t pos = cur_ptr[b]++;
            out_qb.ids[(size_t)pos] = base_id ^ tmp_maskdim[(size_t)i];
        }
    } else {
        for (int i = i0; i < M; ++i) {
            float s = tmp_scores[(size_t)i];
            if (s > limit) continue;
            int b = score_to_bin(s);
            if (b < 0) b = 0;
            if (b >= NBINS) b = NBINS - 1;
            uint32_t end = out_qb.offsets[(size_t)b + 1];
            uint32_t pos = cur_ptr[b];
            if (pos >= end) continue;
            cur_ptr[b] = pos + 1;
            out_qb.ids[(size_t)pos] = base_id ^ tmp_maskdim[(size_t)i];
        }
    }

    return kept_raw;
}

} // namespace qbpg_bins_only_v3a
