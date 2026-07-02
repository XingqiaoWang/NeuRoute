// csr_index_runtime.h
#pragma once
//
// CSR inverted-list index runtime loader (mmap, ZERO-COPY).
//
// Supports BOTH:
//   (A) codes as float32 : codes.float32.csr.bin  (N * d * 4 bytes)
//   (B) codes as uint8   : codes.uint8.csr.bin    (N * d * 1 bytes)
//
// Always maps offsets / bucket_sizes / ids.
// - offsets:      uint64  [nlist+1]
// - bucket_sizes: uint32  [nlist]
// - ids:          int64   [N]
// - codes:        float32 or uint8  [N*d] in row_major
//
// Runtime prepare() computes size stats and builds a 3-tier bucket class:
//   TINY / NORMAL / BIG
//
// Note:
// - This header is standalone: no external deps besides libc/syscalls.
// - All reads are RO-mmap; no allocations proportional to N*d.
//
// Author: updated to support u8 codes.
//

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static inline void csr_die(const std::string& msg) {
    throw std::runtime_error(msg);
}

static inline uint64_t csr_file_size_bytes(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        csr_die("stat failed: " + path + " err=" + std::string(strerror(errno)));
    }
    return (uint64_t)st.st_size;
}

struct CSR_MMapRegion {
    std::string path;
    int fd = -1;
    void* ptr = nullptr;
    uint64_t bytes = 0;

    void map_ro(const std::string& p, int madv = MADV_SEQUENTIAL) {
        path = p;
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) csr_die("open failed: " + path + " err=" + std::string(strerror(errno)));

        bytes = csr_file_size_bytes(path);
        ptr = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            ptr = nullptr;
            ::close(fd);
            fd = -1;
            csr_die("mmap failed: " + path + " err=" + std::string(strerror(errno)));
        }
        (void)::madvise(ptr, bytes, madv);
    }

    void unmap_close() {
        if (ptr) {
            ::munmap(ptr, bytes);
            ptr = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        bytes = 0;
    }

    ~CSR_MMapRegion() { unmap_close(); }
};

// ===============================
//  Bucket stats + threshold policy
// ===============================
struct CSRBucketStats {
    uint32_t nlist = 0;
    uint64_t N = 0;

    uint64_t non_empty = 0;
    uint64_t empty = 0;
    uint64_t sum_sizes = 0;

    uint32_t min_nonempty = 0;
    uint32_t max_size = 0;

    double mean_all = 0.0;
    double mean_nonempty = 0.0;

    // percentiles (non-empty)
    uint32_t p50 = 0;
    uint32_t p90 = 0;
    uint32_t p95 = 0;
    uint32_t p99 = 0;
    uint32_t p999 = 0;
    uint32_t p9995 = 0;

    // selected thresholds
    uint32_t tiny_thr = 0;
    uint32_t big_thr = 0;

    void print(FILE* fp = stderr) const {
        std::fprintf(fp, "==== Bucket size stats ====\n");
        std::fprintf(fp, "nlist          = %u\n", nlist);
        std::fprintf(fp, "non_empty      = %llu (%.6f%%)\n",
                     (unsigned long long)non_empty,
                     100.0 * (double)non_empty / (double)nlist);
        std::fprintf(fp, "sum_sizes      = %llu (should equal N=%llu)\n",
                     (unsigned long long)sum_sizes,
                     (unsigned long long)N);
        std::fprintf(fp, "min_nonempty   = %u\n", min_nonempty);
        std::fprintf(fp, "max            = %u\n", max_size);
        std::fprintf(fp, "mean_all       = %.6f\n", mean_all);
        std::fprintf(fp, "mean_nonempty  = %.6f\n", mean_nonempty);
        std::fprintf(fp, "==========================\n");

        std::fprintf(fp, "Percentiles over NON-EMPTY buckets:\n");
        std::fprintf(fp, "  p50=%u p90=%u p95=%u p99=%u p999=%u p9995=%u\n",
                     p50, p90, p95, p99, p999, p9995);

        std::fprintf(fp, "Chosen thresholds:\n");
        std::fprintf(fp, "  tiny_thr=%u  big_thr=%u (big=p9995)\n", tiny_thr, big_thr);
        std::fprintf(fp, "==========================\n");
    }
};

// ===============================
//  3-tier bucket class (tiny/normal/big)
// ===============================
struct BucketClass3Tier {
    enum : uint8_t { NORMAL = 0, TINY = 1, BIG = 2 };

    uint32_t nlist = 0;
    uint32_t tiny_thr = 0;
    uint32_t big_thr = 0;
    bool empty_as_tiny = true;

    std::vector<uint8_t> klass; // size nlist, value in {0,1,2}

    // counts + workload sums
    uint64_t cnt_tiny = 0, cnt_normal = 0, cnt_big = 0;
    uint64_t vec_tiny = 0, vec_normal = 0, vec_big = 0;

    inline bool is_normal(uint32_t lid) const { return klass[lid] == NORMAL; }
    inline bool is_big(uint32_t lid) const { return klass[lid] == BIG; }
    inline bool is_tiny(uint32_t lid) const { return klass[lid] == TINY; }

    inline uint8_t k(uint32_t lid) const { return klass[lid]; }

    static inline void print_ratio(
        const char* name,
        uint64_t bcnt, uint64_t vsum,
        uint64_t total_buckets, uint64_t total_vecs
    ) {
        double pb = total_buckets ? 100.0 * (double)bcnt / (double)total_buckets : 0.0;
        double pv = total_vecs ? 100.0 * (double)vsum / (double)total_vecs : 0.0;
        double mean = bcnt ? (double)vsum / (double)bcnt : 0.0;
        std::fprintf(stderr, "  [%s] buckets=%llu (%.6f%%)  vecs=%llu (%.6f%%)  mean=%.3f\n",
                     name,
                     (unsigned long long)bcnt, pb,
                     (unsigned long long)vsum, pv,
                     mean);
    }

    void print(uint64_t N_total) const {
        std::fprintf(stderr, "==== 3-tier bucket class ====\n");
        std::fprintf(stderr, "nlist=%u  tiny_thr=%u  big_thr=%u  empty_as_tiny=%d\n",
                     nlist, tiny_thr, big_thr, (int)empty_as_tiny);
        print_ratio("TINY(+empty)", cnt_tiny, vec_tiny, nlist, N_total);
        print_ratio("NORMAL",       cnt_normal, vec_normal, nlist, N_total);
        print_ratio("BIG",          cnt_big, vec_big, nlist, N_total);
        std::fprintf(stderr, "=============================\n");
    }
};

// ===============================
//  CSR index (mmap) + runtime prepare()
// ===============================
struct CSRIndexMMap {
    // meta
    uint32_t d = 0;
    uint64_t N = 0;
    uint32_t nlist = 0;

    // codes dtype
    bool codes_is_u8 = false;

    // paths
    std::string path_codes, path_ids, path_offsets, path_bucket_sizes, path_big_bucket_ids;

    // mmaps
    CSR_MMapRegion mm_offsets, mm_bucket_sizes, mm_ids, mm_codes, mm_big;

    // typed ptrs
    const uint64_t* offsets = nullptr;      // [nlist+1]
    const uint32_t* bucket_sizes = nullptr; // [nlist]
    const int64_t*  ids = nullptr;          // [N]

    // codes:
    // - if codes_is_u8=false : codes_f32 is valid, codes_u8=null
    // - if codes_is_u8=true  : codes_u8 is valid, codes_f32=null
    const float*    codes_f32 = nullptr;    // [N*d]
    const uint8_t*  codes_u8  = nullptr;    // [N*d]

    // optional big id list
    const int32_t*  big_bucket_ids = nullptr;
    uint64_t big_count = 0;

    // runtime prepared objects
    CSRBucketStats stats;
    BucketClass3Tier bucket_class;

    // ------------------------------------------------------------
    // open(): mmap all files, validate sizes, and optionally prepare()
    //
    // codes_is_u8_:
    //   false => codes file is float32 (N*d*4 bytes)
    //   true  => codes file is uint8  (N*d*1 bytes)
    // ------------------------------------------------------------
    void open(
        uint32_t d_,
        uint64_t N_,
        uint32_t nlist_,
        const std::string& codes_path,
        const std::string& ids_path,
        const std::string& offsets_path,
        const std::string& bucket_sizes_path,
        const std::string& big_bucket_ids_path = "",
        bool do_prepare = true,
        bool codes_is_u8_ = false
    ) {
        d = d_;
        N = N_;
        nlist = nlist_;
        codes_is_u8 = codes_is_u8_;

        path_codes = codes_path;
        path_ids = ids_path;
        path_offsets = offsets_path;
        path_bucket_sizes = bucket_sizes_path;
        path_big_bucket_ids = big_bucket_ids_path;

        // Map required files
        // offsets/bucket_sizes sequential scan is fine.
        // ids/codes are random access in search => MADV_RANDOM helps.
        mm_offsets.map_ro(path_offsets, MADV_SEQUENTIAL);
        mm_bucket_sizes.map_ro(path_bucket_sizes, MADV_SEQUENTIAL);
        mm_ids.map_ro(path_ids, MADV_RANDOM);
        mm_codes.map_ro(path_codes, MADV_RANDOM);

        offsets = (const uint64_t*)mm_offsets.ptr;
        bucket_sizes = (const uint32_t*)mm_bucket_sizes.ptr;
        ids = (const int64_t*)mm_ids.ptr;

        // Codes pointer depends on dtype
        if (!codes_is_u8) {
            codes_f32 = (const float*)mm_codes.ptr;
            codes_u8 = nullptr;
        } else {
            codes_u8 = (const uint8_t*)mm_codes.ptr;
            codes_f32 = nullptr;
        }

        // Optional big bucket id list
        if (!path_big_bucket_ids.empty()) {
            mm_big.map_ro(path_big_bucket_ids, MADV_SEQUENTIAL);
            big_bucket_ids = (const int32_t*)mm_big.ptr;
            big_count = mm_big.bytes / sizeof(int32_t);
        } else {
            big_bucket_ids = nullptr;
            big_count = 0;
        }

        // -------- size checks --------
        {
            uint64_t expect = (uint64_t)(nlist + 1) * sizeof(uint64_t);
            if (mm_offsets.bytes != expect) csr_die("offsets bytes mismatch");
        }
        {
            uint64_t expect = (uint64_t)(nlist) * sizeof(uint32_t);
            if (mm_bucket_sizes.bytes != expect) csr_die("bucket_sizes bytes mismatch");
        }
        {
            uint64_t expect = (uint64_t)N * sizeof(int64_t);
            if (mm_ids.bytes != expect) csr_die("ids bytes mismatch");
        }
        {
            uint64_t elem = codes_is_u8 ? sizeof(uint8_t) : sizeof(float);
            uint64_t expect = (uint64_t)N * (uint64_t)d * elem;
            if (mm_codes.bytes != expect) {
                csr_die("codes bytes mismatch (check dtype + d + N)");
            }
        }

        // -------- sanity --------
        if (!offsets) csr_die("offsets mmap failed");
        if (!bucket_sizes) csr_die("bucket_sizes mmap failed");
        if (!ids) csr_die("ids mmap failed");
        if (!mm_codes.ptr) csr_die("codes mmap failed");

        if (offsets[0] != 0) csr_die("offsets[0] must be 0");
        if (offsets[nlist] != N) csr_die("offsets[nlist] must equal N");

        // Optional runtime prepare (stats + thresholds + 3-tier)
        if (do_prepare) {
            // Default policy:
            //   big = p9995(non-empty)
            //   tiny = max(p50(non-empty), 10)
            prepare_runtime(
                /*big_quantile=*/0.9995,
                /*empty_as_tiny=*/true,
                /*tiny_min_thr=*/10,
                /*tiny_use_auto_pow2=*/false,
                /*verbose=*/true
            );
        }
    }

    // Range in [0..N] for list_id
    inline std::pair<uint64_t,uint64_t> range(uint32_t list_id) const {
        if (list_id >= nlist) csr_die("list_id out of range");
        return { offsets[list_id], offsets[list_id + 1] };
    }

    inline uint32_t size(uint32_t list_id) const {
        if (list_id >= nlist) csr_die("list_id out of range");
        return bucket_sizes[list_id];
    }

    inline const int64_t* list_ids_ptr(uint32_t list_id) const {
        auto [a,b] = range(list_id); (void)b;
        return ids + a;
    }

    // -------- codes pointer helpers --------

    // Float32 codes pointer (valid only if codes_is_u8=false)
    inline const float* list_codes_ptr_f32(uint32_t list_id) const {
        if (codes_is_u8) csr_die("list_codes_ptr_f32() called but codes_is_u8=true");
        auto [a,b] = range(list_id); (void)b;
        return codes_f32 + (uint64_t)a * (uint64_t)d;
    }

    // UInt8 codes pointer (valid only if codes_is_u8=true)
    inline const uint8_t* list_codes_ptr_u8(uint32_t list_id) const {
        if (!codes_is_u8) csr_die("list_codes_ptr_u8() called but codes_is_u8=false");
        auto [a,b] = range(list_id); (void)b;
        return codes_u8 + (uint64_t)a * (uint64_t)d;
    }

    // Convenience: bytes for one vector
    inline uint32_t vec_bytes() const {
        return (uint32_t)d * (uint32_t)(codes_is_u8 ? sizeof(uint8_t) : sizeof(float));
    }

    // ---------------------------
    // runtime prepare: stats + thresholds + 3-tier class
    // ---------------------------

    static inline uint32_t pick_quantile_nonempty_nth(
        const uint32_t* sizes,
        uint32_t nlist,
        double q
    ) {
        if (q < 0.0) q = 0.0;
        if (q > 1.0) q = 1.0;

        std::vector<uint32_t> v;
        v.reserve(nlist);
        for (uint32_t i = 0; i < nlist; ++i) {
            uint32_t s = sizes[i];
            if (s > 0) v.push_back(s);
        }
        if (v.empty()) return 0;

        size_t m = v.size();
        double pos = q * (double)(m - 1);
        size_t k = (size_t)(pos + 0.5);
        if (k >= m) k = m - 1;

        std::nth_element(v.begin(), v.begin() + (ptrdiff_t)k, v.end());
        return v[k];
    }

    // OPTIONAL old policy:
    // Choose smallest power-of-two t such that:
    //   bucket_frac(size<=t) >= target_bucket_frac
    //   vec_frac(size<=t)    <= max_vec_frac
    static inline uint32_t choose_tiny_thr_auto_pow2(
        const uint32_t* sizes,
        uint32_t nlist,
        uint64_t N,
        bool empty_as_tiny,
        double target_bucket_frac = 0.70,
        double max_vec_frac = 0.15
    ) {
        const uint32_t cand[] = {8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
        const int nc = (int)(sizeof(cand)/sizeof(cand[0]));

        for (int ci = 0; ci < nc; ++ci) {
            uint32_t t = cand[ci];
            uint64_t bcnt = 0;
            uint64_t vsum = 0;

            for (uint32_t i = 0; i < nlist; ++i) {
                uint32_t s = sizes[i];
                if (s == 0) {
                    if (empty_as_tiny) bcnt++;
                    continue;
                }
                if (s <= t) {
                    bcnt++;
                    vsum += (uint64_t)s;
                }
            }

            double bf = (double)bcnt / (double)nlist;
            double vf = N ? (double)vsum / (double)N : 0.0;

            if (bf >= target_bucket_frac && vf <= max_vec_frac) return t;
        }
        return 128;
    }

    // NEW default tiny threshold policy:
    // tiny_thr = max(p50(non-empty), tiny_min_thr)
    static inline uint32_t choose_tiny_thr_p50_floor(
        uint32_t p50_nonempty,
        uint32_t tiny_min_thr
    ) {
        if (tiny_min_thr < 1) tiny_min_thr = 1;
        return (p50_nonempty > tiny_min_thr) ? p50_nonempty : tiny_min_thr;
    }

    void prepare_runtime(
        double big_quantile = 0.9995,   // p9995(non-empty)
        bool empty_as_tiny = true,

        // tiny policy knobs (NEW default)
        uint32_t tiny_min_thr = 10,
        bool tiny_use_auto_pow2 = false,

        // optional auto_pow2 knobs
        double target_bucket_frac_for_tiny = 0.70,
        double max_vec_frac_for_tiny = 0.15,

        bool verbose = true
    ) {
        if (!bucket_sizes) csr_die("prepare_runtime() requires bucket_sizes mapped");

        // fill basic stats
        stats = CSRBucketStats{};
        stats.nlist = nlist;
        stats.N = N;

        uint64_t non_empty = 0, sum = 0;
        uint32_t mn = UINT32_MAX, mx = 0;

        for (uint32_t i = 0; i < nlist; ++i) {
            uint32_t s = bucket_sizes[i];
            sum += (uint64_t)s;
            if (s > 0) {
                non_empty++;
                mn = std::min(mn, s);
                mx = std::max(mx, s);
            }
        }

        stats.non_empty = non_empty;
        stats.empty = (uint64_t)nlist - non_empty;
        stats.sum_sizes = sum;
        stats.min_nonempty = (non_empty ? mn : 0);
        stats.max_size = mx;
        stats.mean_all = (double)sum / (double)nlist;
        stats.mean_nonempty = non_empty ? (double)sum / (double)non_empty : 0.0;

        // percentiles
        stats.p50   = pick_quantile_nonempty_nth(bucket_sizes, nlist, 0.50);
        stats.p90   = pick_quantile_nonempty_nth(bucket_sizes, nlist, 0.90);
        stats.p95   = pick_quantile_nonempty_nth(bucket_sizes, nlist, 0.95);
        stats.p99   = pick_quantile_nonempty_nth(bucket_sizes, nlist, 0.99);
        stats.p999  = pick_quantile_nonempty_nth(bucket_sizes, nlist, 0.999);
        stats.p9995 = pick_quantile_nonempty_nth(bucket_sizes, nlist, 0.9995);

        // big threshold
        stats.big_thr = pick_quantile_nonempty_nth(bucket_sizes, nlist, big_quantile);

        // tiny threshold
        if (!tiny_use_auto_pow2) {
            stats.tiny_thr = choose_tiny_thr_p50_floor(stats.p50, tiny_min_thr);
        } else {
            stats.tiny_thr = choose_tiny_thr_auto_pow2(
                bucket_sizes, nlist, N, empty_as_tiny,
                target_bucket_frac_for_tiny,
                max_vec_frac_for_tiny
            );
        }

        // ensure disjoint (big > tiny)
        if (stats.big_thr <= stats.tiny_thr + 1) {
            stats.big_thr = stats.tiny_thr + 2;
        }

        // build bucket_class
        bucket_class = BucketClass3Tier{};
        bucket_class.nlist = nlist;
        bucket_class.tiny_thr = stats.tiny_thr;
        bucket_class.big_thr = stats.big_thr;
        bucket_class.empty_as_tiny = empty_as_tiny;
        bucket_class.klass.assign(nlist, BucketClass3Tier::NORMAL);

        uint64_t cnt_tiny=0,cnt_normal=0,cnt_big=0;
        uint64_t vec_tiny=0,vec_normal=0,vec_big=0;

        for (uint32_t i = 0; i < nlist; ++i) {
            uint32_t s = bucket_sizes[i];

            if (s == 0) {
                if (empty_as_tiny) {
                    bucket_class.klass[i] = BucketClass3Tier::TINY;
                    cnt_tiny++;
                } else {
                    bucket_class.klass[i] = BucketClass3Tier::NORMAL;
                    cnt_normal++;
                }
                continue;
            }

            if (s >= stats.big_thr) {
                bucket_class.klass[i] = BucketClass3Tier::BIG;
                cnt_big++;
                vec_big += (uint64_t)s;
            } else if (s <= stats.tiny_thr) {
                bucket_class.klass[i] = BucketClass3Tier::TINY;
                cnt_tiny++;
                vec_tiny += (uint64_t)s;
            } else {
                bucket_class.klass[i] = BucketClass3Tier::NORMAL;
                cnt_normal++;
                vec_normal += (uint64_t)s;
            }
        }

        bucket_class.cnt_tiny = cnt_tiny;
        bucket_class.cnt_normal = cnt_normal;
        bucket_class.cnt_big = cnt_big;
        bucket_class.vec_tiny = vec_tiny;
        bucket_class.vec_normal = vec_normal;
        bucket_class.vec_big = vec_big;

        if (verbose) {
            stats.print(stderr);
            bucket_class.print(N);
        }
    }
};


// =====================================================================================
// Clustered Sub-CSR runtime (bucket -> cluster -> ids/codes), produced by bucket_cluster v6.7+
//
// Files under subcsr_dir (mmap, RO):
//   sub_offsets.u64.bin           : uint64 [n_sub+1]
//   sub_ids.i64.bin               : int64  [N]
//   sub_bucket_id.u32.bin         : uint32 [n_sub]
//   sub_cluster_id.u16.bin|u32    : uint16/uint32 [n_sub]
//   sub_centroids.f32.bin         : float32 [n_sub, D]
//   bucket_sub_offsets.u64.bin    : uint64 [nlist+1]  (maps bucket -> sub range)
//
// Optional codes aligned to sub_ids order (row-major):
//   codes.float32.subcsr.bin      : float32 [N, Dcode]  (or D)
//   codes.uint8.subcsr.bin        : uint8   [N, Dcode]
//
// Bucket class policy (by cluster count k per bucket):
//   k == 0                    => TINY (no clusters; bucket not clustered)
//   1 <= k <= small_k_thr     => SMALL
//   small_k_thr < k < big_k   => NORMAL
//   k >= big_k_thr            => BIG
//
// Note: This runtime does NOT assume bucketed_vectors/bucketed_ids exist.
// It only needs subcsr/* plus optional codes aligned to sub_ids order.
// =====================================================================================

enum class CSRBucketClass4 : uint8_t { TINY=0, SMALL=1, NORMAL=2, BIG=3 };

struct CSRBucketClass4Info {
    std::vector<CSRBucketClass4> klass; // size nlist
    uint64_t cnt_tiny=0, cnt_small=0, cnt_normal=0, cnt_big=0;

    void print(FILE* fp = stderr) const {
        fprintf(fp, "Bucket classes: tiny=%llu small=%llu normal=%llu big=%llu\n",
                (unsigned long long)cnt_tiny,
                (unsigned long long)cnt_small,
                (unsigned long long)cnt_normal,
                (unsigned long long)cnt_big);
    }
};

struct CSRSubCSRIndexRuntime {
    // dims
    uint64_t N = 0;          // number of vectors (sub_ids length)
    uint32_t D = 0;          // centroid dim
    uint32_t Dcode = 0;      // code dim (optional; 0 => none)
    uint64_t nlist = 0;
    uint64_t n_sub = 0;

    // class thresholds
    uint32_t big_k_thr   = 1000;
    uint32_t small_k_thr = 1;

    // code mode
    bool has_codes = false;
    bool codes_is_u8 = false;

    // mmaps
    CSR_MMapRegion mm_sub_offsets;
    CSR_MMapRegion mm_sub_ids;
    CSR_MMapRegion mm_sub_bucket_id;
    CSR_MMapRegion mm_sub_cluster_id;
    CSR_MMapRegion mm_sub_centroids;
    CSR_MMapRegion mm_bucket_sub_offsets;
    CSR_MMapRegion mm_codes; // optional

    // typed pointers
    const uint64_t* sub_offsets = nullptr;          // [n_sub+1]
    const int64_t*  sub_ids     = nullptr;          // [N]
    const uint32_t* sub_bucket  = nullptr;          // [n_sub]
    const void*     sub_cluster = nullptr;          // [n_sub] u16/u32
    bool            sub_cluster_is_u16 = true;
    const float*    sub_centroids = nullptr;        // [n_sub, D]
    const uint64_t* bucket_sub_offsets = nullptr;   // [nlist+1]

    const float* codes_f32 = nullptr;               // [N, Dcode] if has_codes && !u8
    const uint8_t* codes_u8 = nullptr;              // [N, Dcode] if has_codes && u8

    CSRBucketClass4Info bucket_class;

    // ---------- helpers ----------
    inline uint64_t bucket_k(uint32_t b) const {
        return bucket_sub_offsets[b+1] - bucket_sub_offsets[b];
    }
    inline std::pair<uint64_t,uint64_t> bucket_sub_range(uint32_t b) const {
        return { bucket_sub_offsets[b], bucket_sub_offsets[b+1] };
    }
    inline std::pair<uint64_t,uint64_t> sub_range(uint64_t s) const {
        return { sub_offsets[s], sub_offsets[s+1] };
    }
    inline uint32_t sub_cluster_id(uint64_t s) const {
        return sub_cluster_is_u16 ? (uint32_t)((const uint16_t*)sub_cluster)[s]
                                  : ((const uint32_t*)sub_cluster)[s];
    }
    inline const float* sub_centroid_ptr(uint64_t s) const {
        return sub_centroids + (size_t)s * D;
    }
    inline const int64_t* sub_ids_ptr(uint64_t s) const {
        auto r = sub_range(s);
        return sub_ids + r.first;
    }
    inline uint64_t sub_size(uint64_t s) const {
        auto r = sub_range(s);
        return r.second - r.first;
    }
    inline const float* sub_codes_f32_ptr(uint64_t s) const {
        if (!has_codes || codes_is_u8) return nullptr;
        auto r = sub_range(s);
        return codes_f32 + (size_t)r.first * Dcode;
    }
    inline const uint8_t* sub_codes_u8_ptr(uint64_t s) const {
        if (!has_codes || !codes_is_u8) return nullptr;
        auto r = sub_range(s);
        return codes_u8 + (size_t)r.first * Dcode;
    }

    // ---------- load ----------
    void load_subcsr(const std::string& subcsr_dir,
                     uint32_t D_centroid,
                     uint64_t nlist_,
                     bool sub_cluster_u16 = true,
                     uint32_t big_k_thr_ = 1000,
                     uint32_t small_k_thr_ = 1,
                     // codes (optional)
                     const std::string& codes_path = "",
                     uint32_t D_code = 0,
                     bool codes_u8_ = false,
                     int madv = MADV_SEQUENTIAL)
    {
        D = D_centroid;
        nlist = (uint32_t)nlist_;
        big_k_thr = big_k_thr_;
        small_k_thr = small_k_thr_;
        sub_cluster_is_u16 = sub_cluster_u16;

        mm_sub_offsets.map_ro(subcsr_dir + "/sub_offsets.u64.bin", madv);
        sub_offsets = (const uint64_t*)mm_sub_offsets.ptr;
        n_sub = (uint64_t)(mm_sub_offsets.bytes / sizeof(uint64_t)) - 1;

        mm_sub_ids.map_ro(subcsr_dir + "/sub_ids.i64.bin", madv);
        sub_ids = (const int64_t*)mm_sub_ids.ptr;
        N = (uint64_t)(mm_sub_ids.bytes / sizeof(int64_t));

        mm_sub_bucket_id.map_ro(subcsr_dir + "/sub_bucket_id.u32.bin", madv);
        sub_bucket = (const uint32_t*)mm_sub_bucket_id.ptr;

        mm_sub_cluster_id.map_ro(subcsr_dir + std::string(sub_cluster_is_u16 ? "/sub_cluster_id.u16.bin"
                                                                             : "/sub_cluster_id.u32.bin"), madv);
        sub_cluster = mm_sub_cluster_id.ptr;

        mm_sub_centroids.map_ro(subcsr_dir + "/sub_centroids.f32.bin", madv);
        sub_centroids = (const float*)mm_sub_centroids.ptr;

        mm_bucket_sub_offsets.map_ro(subcsr_dir + "/bucket_sub_offsets.u64.bin", madv);
        bucket_sub_offsets = (const uint64_t*)mm_bucket_sub_offsets.ptr;

        // basic consistency checks (lightweight)
        if (bucket_sub_offsets == nullptr) csr_die("bucket_sub_offsets null");
        if ((uint64_t)(mm_bucket_sub_offsets.bytes / sizeof(uint64_t)) != (uint64_t)nlist_ + 1) {
            csr_die("bucket_sub_offsets length mismatch (expected nlist+1)");
        }
        if (sub_offsets[n_sub] != N) {
            csr_die("sub_offsets last != N (corrupt subcsr?)");
        }
        // centroids size
        uint64_t expect_cent = (uint64_t)n_sub * (uint64_t)D * sizeof(float);
        if (mm_sub_centroids.bytes != expect_cent) {
            csr_die("sub_centroids size mismatch (expected n_sub*D*f32)");
        }

        // optional codes
        has_codes = false;
        codes_is_u8 = false;
        Dcode = 0;
        codes_f32 = nullptr;
        codes_u8 = nullptr;
        if (!codes_path.empty()) {
            if (D_code == 0) csr_die("codes_path given but D_code==0");
            Dcode = D_code;
            has_codes = true;
            codes_is_u8 = codes_u8_;
            mm_codes.map_ro(codes_path, madv);
            uint64_t expect = (uint64_t)N * (uint64_t)Dcode * (codes_is_u8 ? 1ull : 4ull);
            if (mm_codes.bytes != expect) {
                csr_die("codes file size mismatch vs N*Dcode");
            }
            if (codes_is_u8) codes_u8 = (const uint8_t*)mm_codes.ptr;
            else             codes_f32 = (const float*)mm_codes.ptr;
        }

        // classify buckets by k
        bucket_class = CSRBucketClass4Info{};
        bucket_class.klass.resize(nlist_);
        for (uint32_t b = 0; b < nlist_; ++b) {
            uint64_t k = bucket_k(b);
            if (k == 0) {
                bucket_class.klass[b] = CSRBucketClass4::TINY; bucket_class.cnt_tiny++;
            } else if (k <= (uint64_t)small_k_thr) {
                bucket_class.klass[b] = CSRBucketClass4::SMALL; bucket_class.cnt_small++;
            } else if (k >= (uint64_t)big_k_thr) {
                bucket_class.klass[b] = CSRBucketClass4::BIG; bucket_class.cnt_big++;
            } else {
                bucket_class.klass[b] = CSRBucketClass4::NORMAL; bucket_class.cnt_normal++;
            }
        }
    }


    // -------------------------------------------------------------------------
    // NEW (v2->v3): load Sub-CSR when "member arrays" (sub_ids/sub_offsets/optional
    // sub_codes) have been materialized into a separate *cluster-CSR* directory,
    // while the "meta" mapping files remain in the original subcsr_dir.
    //
    // This matches your current pipeline outputs:
    //   meta_dir (from bucket_cluster --out_sub_dir):
    //     sub_bucket_id.u32.bin
    //     sub_cluster_id.u16.bin|u32.bin
    //     sub_centroids.f32.bin          (n_sub, D)
    //     bucket_sub_offsets.u64.bin     (nlist+1)
    //
    //   cluster_csr_dir (from csr_build_v2 cluster mode):
    //     sub_offsets.uint64.bin         (n_sub+1)   [raw u64]
    //     sub_ids.int64.csr.bin          (N)         [raw i64]
    //     sub_codes.<dtype>.csr.bin      (N, Dcode)  [optional]
    //     sub_sizes.bin                  (n_sub)     [optional; not required here]
    //     big_sub_ids.bin                (optional; not required here)
    //
    // NOTE:
    // - We intentionally do NOT depend on sub_sizes/big_sub_ids: runtime can
    //   compute per-sub size from sub_offsets.
    // - ids are still required to map local positions back to global ids.
    // -------------------------------------------------------------------------
    void load_subcsr_split(
        const std::string& meta_dir,
        const std::string& cluster_csr_dir,
        uint32_t D_centroid,
        uint64_t nlist_,
        bool sub_cluster_u16 = true,
        uint32_t big_k_thr_ = 1000,
        uint32_t small_k_thr_ = 1,
        // optional sub_codes under cluster_csr_dir
        const std::string& sub_codes_path = "",
        uint32_t D_code = 0,
        bool codes_u8_ = false,
        int madv_meta = MADV_SEQUENTIAL,
        int madv_data = MADV_RANDOM
    ) {
        D = D_centroid;
        nlist = (uint32_t)nlist_;
        big_k_thr = big_k_thr_;
        small_k_thr = small_k_thr_;
        sub_cluster_is_u16 = sub_cluster_u16;

        // ---- member arrays (cluster-CSR outputs) ----
        {
            std::string p_off = cluster_csr_dir + "/sub_offsets.uint64.bin";
            std::string p_ids = cluster_csr_dir + "/sub_ids.int64.csr.bin";
            mm_sub_offsets.map_ro(p_off, madv_meta);
            sub_offsets = (const uint64_t*)mm_sub_offsets.ptr;
            n_sub = (uint64_t)(mm_sub_offsets.bytes / sizeof(uint64_t)) - 1;

            mm_sub_ids.map_ro(p_ids, madv_data);
            sub_ids = (const int64_t*)mm_sub_ids.ptr;
            N = (uint64_t)(mm_sub_ids.bytes / sizeof(int64_t));

            if (sub_offsets[n_sub] != N) {
                csr_die("sub_offsets last != N (cluster-CSR mismatch?)");
            }
        }

        // ---- meta mapping files (original subcsr) ----
        mm_sub_bucket_id.map_ro(meta_dir + "/sub_bucket_id.u32.bin", madv_meta);
        sub_bucket = (const uint32_t*)mm_sub_bucket_id.ptr;

        mm_sub_cluster_id.map_ro(meta_dir + std::string(sub_cluster_is_u16 ? "/sub_cluster_id.u16.bin"
                                                                           : "/sub_cluster_id.u32.bin"), madv_meta);
        sub_cluster = mm_sub_cluster_id.ptr;

        mm_sub_centroids.map_ro(meta_dir + "/sub_centroids.f32.bin", madv_meta);
        sub_centroids = (const float*)mm_sub_centroids.ptr;

        mm_bucket_sub_offsets.map_ro(meta_dir + "/bucket_sub_offsets.u64.bin", madv_meta);
        bucket_sub_offsets = (const uint64_t*)mm_bucket_sub_offsets.ptr;

        // ---- consistency checks ----
        if ((uint64_t)(mm_bucket_sub_offsets.bytes / sizeof(uint64_t)) != (uint64_t)nlist_ + 1) {
            csr_die("bucket_sub_offsets length mismatch (expected nlist+1)");
        }
        // centroids size must match n_sub
        {
            uint64_t expect_cent = (uint64_t)n_sub * (uint64_t)D * sizeof(float);
            if (mm_sub_centroids.bytes != expect_cent) {
                csr_die("sub_centroids size mismatch (expected n_sub*D*f32)");
            }
        }
        // sub_bucket/sub_cluster arrays must match n_sub
        {
            uint64_t expect_b = (uint64_t)n_sub * sizeof(uint32_t);
            if (mm_sub_bucket_id.bytes != expect_b) {
                csr_die("sub_bucket_id size mismatch (expected n_sub*u32)");
            }
            uint64_t expect_c = (uint64_t)n_sub * (sub_cluster_is_u16 ? sizeof(uint16_t) : sizeof(uint32_t));
            if (mm_sub_cluster_id.bytes != expect_c) {
                csr_die("sub_cluster_id size mismatch (expected n_sub*u16/u32)");
            }
        }

        // ---- optional sub_codes ----
        has_codes = false;
        codes_is_u8 = false;
        Dcode = 0;
        codes_f32 = nullptr;
        codes_u8 = nullptr;

        if (!sub_codes_path.empty()) {
            if (D_code == 0) csr_die("sub_codes_path given but D_code==0");
            Dcode = D_code;
            has_codes = true;
            codes_is_u8 = codes_u8_;
            mm_codes.map_ro(sub_codes_path, madv_data);
            uint64_t expect = (uint64_t)N * (uint64_t)Dcode * (codes_is_u8 ? 1ull : 4ull);
            if (mm_codes.bytes != expect) csr_die("sub_codes file size mismatch vs N*Dcode");
            if (codes_is_u8) codes_u8 = (const uint8_t*)mm_codes.ptr;
            else             codes_f32 = (const float*)mm_codes.ptr;
        }

        // ---- classify buckets by cluster count k ----
        bucket_class = CSRBucketClass4Info{};
        bucket_class.klass.resize(nlist_);
        for (uint32_t b = 0; b < nlist_; ++b) {
            uint64_t k = bucket_k(b);
            if (k == 0) {
                bucket_class.klass[b] = CSRBucketClass4::TINY; bucket_class.cnt_tiny++;
            } else if (k <= (uint64_t)small_k_thr) {
                bucket_class.klass[b] = CSRBucketClass4::SMALL; bucket_class.cnt_small++;
            } else if (k >= (uint64_t)big_k_thr) {
                bucket_class.klass[b] = CSRBucketClass4::BIG; bucket_class.cnt_big++;
            } else {
                bucket_class.klass[b] = CSRBucketClass4::NORMAL; bucket_class.cnt_normal++;
            }
        }
    }

    void unmap_close() {
        mm_sub_offsets.unmap_close();
        mm_sub_ids.unmap_close();
        mm_sub_bucket_id.unmap_close();
        mm_sub_cluster_id.unmap_close();
        mm_sub_centroids.unmap_close();
        mm_bucket_sub_offsets.unmap_close();
        mm_codes.unmap_close();

        sub_offsets = nullptr;
        sub_ids = nullptr;
        sub_bucket = nullptr;
        sub_cluster = nullptr;
        sub_centroids = nullptr;
        bucket_sub_offsets = nullptr;
        codes_f32 = nullptr;
        codes_u8 = nullptr;
        bucket_class = CSRBucketClass4Info{};
        N = 0; D = 0; Dcode = 0; nlist = 0; n_sub = 0;
        has_codes = false; codes_is_u8 = false;
    }
};

// =====================================================================================

