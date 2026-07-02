// bucket_cluster_pipeline_v6_8_1_gt.cpp
//
// Bucket-level k-means clustering + optional GT analysis.
//
// This is an UPDATED full version (English comments) based on your v6.5_gt draft,
// with the following key upgrades:
//
// (A) Small-bucket clustering
//     - Buckets with 32 <= sz < 128 are also clustered.
//     - k_small = ceil(sz / small_cluster_target) (default target=32)
//     - Controlled by:
//         --min_small_bucket_cluster  (default 32)
//         --small_cluster_target      (default 32)
//     - Large bucket clustering remains:
//         sz >= --min_bucket_cluster  (default 129) => k = ceil(sz / cluster_target)
//
// (B) Optional GT analysis (reads exactly your bin format)
//     - gt_I: int64 contiguous [Q,K] written by numpy tofile()
//         gt_I.int64.Q{Q}.K{K}.bin
//     - gt_D: float32 contiguous [Q,K] written by numpy tofile()
//         gt_D.float32.Q{Q}.K{K}.bin
//       IMPORTANT: by default we assume gt_D stores L2^2 (squared L2).
//         --gtD_is_l2sqr 1  (default)
//         --gtD_is_l2sqr 0  (if your gt_D is L2)
//
//     - Query vectors (compressed): --q_npy <float32 [Q,D]>
//
//     It compares (per gt pair):
//       d_q_gt_recompute_{l2,l2sqr}  recomputed from xb
//       d_q_gt_given_{l2,l2sqr}      derived from gt_D (using gtD_is_l2sqr)
//       d_q_cent_{l2,l2sqr}          distance from q to centroid(cluster(id))
//
//     It outputs:
//       - gt_cluster_distance_pairs.csv (optional) with BOTH domains + given/recompute
//       - gt_cluster_distance_summary.txt with quantiles in L2 and L2^2,
//         plus linear fit / correlation between given and recompute distances.
//
// Build example:
//   g++ -O3 -std=c++17 -fopenmp bucket_cluster_pipeline_v6_8_1_gt.cpp \
//       -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfaiss -lopenblas \
//       -Wl,-rpath,$CONDA_PREFIX/lib -o bucket_cluster_pipeline_v6_7
//
// Notes:
//   - For N~1B, GT analysis builds id2pos (uint32[N]) ~4GB.
//   - If --gt_build_pos2bucket=1, it also builds pos2bucket (uint32[N]) ~4GB.
//   - If memory is tight, set --gt_build_pos2bucket 0 (slower lookup: binary search offsets).
//

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void openblas_set_num_threads(int) __attribute__((weak));

using Clock = std::chrono::high_resolution_clock;

static inline double now_sec() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
static inline double elapsed_sec(double t0) { return now_sec() - t0; }

static void die(const std::string& msg) {
    std::cerr << "[FATAL] " << msg << "\n";
    std::exit(1);
}

static uint64_t file_size(const std::string& path) {
    std::error_code ec;
    auto sz = std::filesystem::file_size(path, ec);
    if (ec) die("file_size failed: " + path + " : " + ec.message());
    return (uint64_t)sz;
}

static void ensure_dir(const std::string& d) {
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    if (ec) die("create_directories failed: " + d + " : " + ec.message());
}

static void copy_file_if_needed(const std::string& src, const std::string& dst) {
    if (std::filesystem::exists(dst)) {
        std::cerr << "[SHM] exists: " << dst << "\n";
        return;
    }
    ensure_dir(std::filesystem::path(dst).parent_path().string());
    std::ifstream in(src, std::ios::binary);
    if (!in) die("open src failed: " + src);
    std::ofstream out(dst, std::ios::binary);
    if (!out) die("open dst failed: " + dst);

    const size_t BUF = 8u << 20;
    std::vector<char> buf(BUF);
    while (in) {
        in.read(buf.data(), (std::streamsize)buf.size());
        std::streamsize got = in.gcount();
        if (got > 0) out.write(buf.data(), got);
    }
    out.flush();
    std::cerr << "[SHM] copied: " << src << " -> " << dst << "\n";
}

struct MMap {
    void*  addr = nullptr;
    size_t len  = 0;
    int    fd   = -1;

    void open_ro(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) die("open_ro failed: " + path + " errno=" + std::to_string(errno));
        struct stat st{};
        if (fstat(fd, &st) != 0) die("fstat failed: " + path);
        len = (size_t)st.st_size;
        addr = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) die("mmap ro failed: " + path + " errno=" + std::to_string(errno));
    }

    void open_rw_create(const std::string& path, size_t bytes) {
        ensure_dir(std::filesystem::path(path).parent_path().string());
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) die("open_rw_create failed: " + path + " errno=" + std::to_string(errno));
        if (ftruncate(fd, (off_t)bytes) != 0) die("ftruncate failed: " + path);
        len = bytes;
        addr = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) die("mmap rw failed: " + path + " errno=" + std::to_string(errno));
    }

    void close_map() {
        if (addr && addr != MAP_FAILED) munmap(addr, len);
        addr = nullptr;
        len = 0;
        if (fd >= 0) ::close(fd);
        fd = -1;
    }

    ~MMap() { close_map(); }
};

// -------------------- Minimal .npy reader (C-order only) --------------------
struct NpyArrayView {
    const void* data = nullptr;
    std::vector<uint64_t> shape;
    std::string descr;
    size_t data_offset = 0;
};

static uint16_t read_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static std::string find_header_string(const std::string& hdr, const std::string& key) {
    auto pos = hdr.find("'" + key + "'");
    if (pos == std::string::npos) return "";
    pos = hdr.find(":", pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < hdr.size() && std::isspace((unsigned char)hdr[pos])) pos++;
    if (pos >= hdr.size()) return "";
    if (hdr[pos] == '\'') {
        size_t end = hdr.find('\'', pos + 1);
        if (end == std::string::npos) return "";
        return hdr.substr(pos + 1, end - (pos + 1));
    }
    return "";
}

static bool find_header_bool(const std::string& hdr, const std::string& key, bool* out) {
    auto pos = hdr.find("'" + key + "'");
    if (pos == std::string::npos) return false;
    pos = hdr.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < hdr.size() && std::isspace((unsigned char)hdr[pos])) pos++;
    if (hdr.compare(pos, 4, "True") == 0) { *out = true; return true; }
    if (hdr.compare(pos, 5, "False") == 0) { *out = false; return true; }
    return false;
}

static bool parse_shape_tuple(const std::string& hdr, std::vector<uint64_t>* shape_out) {
    shape_out->clear();
    auto pos = hdr.find("'shape'");
    if (pos == std::string::npos) return false;
    pos = hdr.find("(", pos);
    if (pos == std::string::npos) return false;
    pos++;

    while (pos < hdr.size()) {
        while (pos < hdr.size() && std::isspace((unsigned char)hdr[pos])) pos++;
        if (pos >= hdr.size()) return false;

        if (hdr[pos] == ')') { pos++; break; }

        if (!std::isdigit((unsigned char)hdr[pos])) return false;
        size_t end = pos;
        while (end < hdr.size() && std::isdigit((unsigned char)hdr[end])) end++;
        uint64_t v = std::stoull(hdr.substr(pos, end - pos));
        shape_out->push_back(v);
        pos = end;

        while (pos < hdr.size() && std::isspace((unsigned char)hdr[pos])) pos++;

        if (pos < hdr.size() && hdr[pos] == ',') { pos++; continue; }
        if (pos < hdr.size() && hdr[pos] == ')') { pos++; break; }

        auto next_rpar = hdr.find(')', pos);
        if (next_rpar == std::string::npos) return false;
        pos = next_rpar + 1;
        break;
    }
    return !shape_out->empty();
}

static NpyArrayView load_npy_view(MMap& mm, const std::string& path) {
    mm.open_ro(path);
    const uint8_t* p = (const uint8_t*)mm.addr;
    if (mm.len < 16) die("npy too small: " + path);

    if (!(p[0] == 0x93 && std::memcmp(p + 1, "NUMPY", 5) == 0))
        die("bad npy magic: " + path);

    uint8_t major = p[6], minor = p[7];
    (void)minor;

    size_t header_len = 0;
    size_t header_off = 0;

    if (major == 1) {
        header_len = read_u16_le(p + 8);
        header_off = 10;
    } else if (major == 2) {
        header_len = (size_t)read_u32_le(p + 8);
        header_off = 12;
    } else {
        die("unsupported npy version: " + std::to_string(major) + "." + std::to_string(minor));
    }
    if (header_off + header_len > mm.len) die("npy header out of range: " + path);

    std::string hdr((const char*)(p + header_off), header_len);

    bool fortran = false;
    if (!find_header_bool(hdr, "fortran_order", &fortran)) die("npy missing fortran_order: " + path);
    if (fortran) die("npy fortran_order True unsupported: " + path);

    std::string descr = find_header_string(hdr, "descr");
    if (descr.empty()) die("npy missing descr: " + path);

    std::vector<uint64_t> shape;
    if (!parse_shape_tuple(hdr, &shape)) die("npy missing/invalid shape: " + path);

    size_t data_off = header_off + header_len;
    if (data_off >= mm.len) die("npy data_off invalid: " + path);

    NpyArrayView v;
    v.data = (const void*)(p + data_off);
    v.shape = std::move(shape);
    v.descr = descr;
    v.data_offset = data_off;
    return v;
}

// -------------------- Stats helpers --------------------
static float percentile_from_sample(std::vector<float>& a, float q) {
    if (a.empty()) return 0.0f;
    q = std::max(0.0f, std::min(1.0f, q));
    size_t k = (size_t)std::floor(q * (a.size() - 1));
    std::nth_element(a.begin(), a.begin() + k, a.end());
    return a[k];
}

struct Reservoir {
    std::vector<float> buf;
    uint64_t seen = 0;
    std::mt19937_64 rng;

    Reservoir(size_t cap, uint64_t seed) : buf() {
        buf.reserve(cap);
        rng.seed(seed);
    }
    inline void push(float x) {
        seen++;
        if (buf.size() < buf.capacity()) {
            buf.push_back(x);
            return;
        }
        std::uniform_int_distribution<uint64_t> dist(0, seen - 1);
        uint64_t j = dist(rng);
        if (j < buf.size()) buf[(size_t)j] = x;
    }
};

// Online accumulators for linear relationship y = a*x + b and correlation.
struct LinAcc {
    uint64_t n = 0;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;

    inline void add(double x, double y) {
        n++;
        sx += x; sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    inline void merge(const LinAcc& o) {
        n += o.n;
        sx += o.sx; sy += o.sy;
        sxx += o.sxx; syy += o.syy;
        sxy += o.sxy;
    }
};

static inline void lin_finalize(const LinAcc& a, double* slope, double* intercept, double* corr, double* rmse) {
    if (a.n < 2) { *slope = 0; *intercept = 0; *corr = 0; *rmse = 0; return; }
    double n = (double)a.n;
    double mx = a.sx / n, my = a.sy / n;
    double vx = a.sxx - n * mx * mx;
    double vy = a.syy - n * my * my;
    double cov = a.sxy - n * mx * my;

    if (vx <= 0) {
        *slope = 0;
        *intercept = my;
        *corr = 0;
        *rmse = 0;
        return;
    }

    *slope = cov / vx;
    *intercept = my - (*slope) * mx;

    if (vx > 0 && vy > 0) *corr = cov / std::sqrt(vx * vy);
    else *corr = 0;

    // SSE for y - (a x + b)
    double sse = a.syy
        + (*slope) * (*slope) * a.sxx
        + n * (*intercept) * (*intercept)
        - 2 * (*slope) * a.sxy
        - 2 * (*intercept) * a.sy
        + 2 * (*slope) * (*intercept) * a.sx;
    sse = std::max(0.0, sse);
    *rmse = std::sqrt(sse / n);
}

static double get_free_gib(const std::string& dir) {
    std::error_code ec;
    auto sp = std::filesystem::space(dir, ec);
    if (ec) return 0.0;
    return (double)sp.available / (1024.0 * 1024.0 * 1024.0);
}

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

static inline float l2sqr(const float* a, const float* b, uint32_t D) {
    float s = 0.0f;
    for (uint32_t d = 0; d < D; d++) {
        float x = a[d] - b[d];
        s += x * x;
    }
    return s;
}
static inline float l2(const float* a, const float* b, uint32_t D) {
    return std::sqrt(std::max(0.0f, l2sqr(a, b, D)));
}

// -------------------- Args --------------------
static void usage() {
    std::cerr <<
R"(Usage:
  bucket_cluster_pipeline_v6_8

VECTORS (choose one):
  --vec_npy <vectors.npy>                              (float32 [N,D])
  OR
  --vec_f32_bin <vectors.f32.bin> --N <u64> --D <u32>  (raw float32 row-major [N,D])

OFFSETS (choose one):
  --offsets_npy <offsets.npy>                          (uint64 [nlist+1], 1D or 2D(n,1))
  OR
  --offsets_u64_bin <offsets.u64.bin> --nlist <u64>    (raw uint64 [nlist+1])

IDS (choose exactly one):
  --indices_u32_bin <indices.u32.bin>                  (uint32 [N])
  OR
  --ids_i64_csr_bin <ids.int64.csr.bin>                (int64 [N]) (must be row-id range [0,N))

  --out_dir <out_dir>

  --out_sub_dir <dir>              (default "") write sub-CSR (bucket->cluster) under this dir
  --sub_cluster_id_u16 <0/1>        (default 1) store sub_cluster_id as uint16 (else uint32)
  --emit_tiny_centroid <0/1>        (default 1) for tiny (unclustered) buckets, store centroid as mean (1) or keep zeros (0)

Optional:
  --threads <int>                (default 24)    outer OpenMP threads
  --blas_threads <int>           (default 1)     openblas threads (if available)
  --stage_shm <0/1>              (default 1)
  --shm_dir <dir>                (default /dev/shm/deep_bucket_cluster)

  --emit_bucketed_vectors <0/1>  (default 1) write bucketed_vectors.f32.bin (very large; saves ~82GiB if 0)
  --emit_bucketed_ids <0/1>      (default 1) write bucketed_ids.u32.bin (saves ~4GiB if 0)

Large-bucket clustering:
  --cluster_target <int>         (default 64)    k = ceil(size / cluster_target)
  --min_bucket_cluster <int>     (default 129)

Small-bucket clustering (NEW):
  --min_small_bucket_cluster <int> (default 32)  enable if sz >= this AND sz < min_bucket_cluster
  --small_cluster_target <int>     (default 32)  k = ceil(size / small_cluster_target)

  --k_cap <int>                  (default 2048)  cap k per bucket
  --sample_cap <int>             (default 20000)
  --kmeans_iter <int>            (default 20)    Lloyd iters on sample
  --full_refine <0/1>            (default 1)
  --full_lloyd_iters <int>       (default 3)
  --reseed_empty_full <0/1>      (default 1)
  --init <kmeanspp|random>       (default kmeanspp)
  --chunk_vecs <int>             (default 4096)

  --do_intra_sep <0/1>           (default 1)
  --do_margin <0/1>              (default 1)
  --margin_cap <int>             (default 20000)
  --margin_eps <float>           (default 0.02)

Disaster retry:
  --retry_disaster <0/1>         (default 1)
  --retry_reduce_k <0/1>         (default 1)
  --overlap_thr <float>          (default 50)
  --intra_min_thr <float>        (default 1e-3)

  --seed <u64>                   (default 12345)

Optional GT analysis (enable by providing required):
  Queries:
    --q_npy <q.npy>                                 (float32 [Q,D])
    OR
    --q_f32_bin <q.f32.bin> --Q <u64> --Dq <u32>    (raw float32 [Q,D])

  GT bins (your format):
    --gt_I_bin <gt_I.int64.Q{Q}.K{K}.bin>           (required when GT enabled)
    --gt_D_bin <gt_D.float32.Q{Q}.K{K}.bin>         (optional)

  Controls:
    --gt_recompute_q_gt <0/1>        (default 1) recompute ||q-xb[id]|| (costly)
    --gt_dump_cap_per_query <int>    (default 0) if >0, dump CSV for first cap per query
    --gt_build_pos2bucket <0/1>      (default 1) build uint32[N] pos->bucket (faster, +4GB)
    --gt_summary_sample <int>        (default 2000000) reservoir sample size for summaries
    --gtD_is_l2sqr <0/1>             (default 1) treat gt_D as L2^2; if 0 treat as L2
    --gt_report_both_domains <0/1>   (default 1) report both L2 and L2^2 summaries

)";
}

struct Args {
    // vectors
    std::string vec_npy;
    std::string vec_f32_bin;
    uint64_t N_arg = 0;
    uint32_t D_arg = 0;

    // offsets
    std::string offsets_npy;
    std::string offsets_u64_bin;
    uint64_t nlist_arg = 0;

    // ids
    std::string indices_u32_bin;
    std::string ids_i64_csr_bin;

    std::string out_dir = "./out";

    // sub-CSR output (bucket -> in-bucket clusters)
    // If empty, sub-CSR is not written.
    // Otherwise, files are written under this directory.
    std::string out_sub_dir = "";

    // store sub_cluster_id as uint16 (else uint32)
    int sub_cluster_id_u16 = 1;

    // For tiny (unclustered) buckets (k_per_bucket==0), whether to compute/store centroid as mean.
    // If 0, we keep the centroid as zeros to save time.
    int emit_tiny_centroid = 1;

    // Whether to emit the very large bucketed vectors/ids reorder files.
    // When emit_bucketed_vectors=0, we do NOT create/mmap bucketed_vectors.f32.bin;
    // instead we gather vectors from xb on-the-fly per bucket during clustering.
    int emit_bucketed_vectors = 1;
    int emit_bucketed_ids = 1;

    int threads = 24;
    int blas_threads = 1;

    int stage_shm = 1;
    std::string shm_dir = "/dev/shm/deep_bucket_cluster";

    // large bucket clustering
    int cluster_target = 64;
    int min_bucket_cluster = 129;

    // small bucket clustering
    int min_small_bucket_cluster = 32;
    int small_cluster_target = 32;

    int k_cap = 2048;

    int sample_cap = 20000;
    int kmeans_iter = 20;
    int full_refine = 1;
    int full_lloyd_iters = 3;
    int reseed_empty_full = 1;

    std::string init = "kmeanspp"; // kmeanspp | random

    int chunk_vecs = 4096;

    int do_intra_sep = 1;
    int do_margin = 1;
    int margin_cap = 20000;
    float margin_eps = 0.02f;

    int retry_disaster = 1;
    int retry_reduce_k = 1;
    float overlap_thr = 50.0f;
    float intra_min_thr = 1e-3f;

    uint64_t seed = 12345;

    // -------- GT analysis --------
    std::string q_npy;
    std::string q_f32_bin;
    uint64_t Q_arg = 0;
    uint32_t Dq_arg = 0;

    std::string gt_I_bin;
    std::string gt_D_bin;

    int gt_recompute_q_gt = 1;
    int gt_dump_cap_per_query = 0;
    int gt_build_pos2bucket = 1;
    int gt_summary_sample = 2000000;

    // NEW: default assume gt_D stores L2^2
    int gtD_is_l2sqr = 1;
    int gt_report_both_domains = 1;
};

static bool eq(const std::string& a, const char* b) { return a == b; }

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* name)->std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };

        if (eq(k, "--vec_npy")) a.vec_npy = need("--vec_npy");
        else if (eq(k, "--vec_f32_bin")) a.vec_f32_bin = need("--vec_f32_bin");
        else if (eq(k, "--N")) a.N_arg = (uint64_t)std::stoull(need("--N"));
        else if (eq(k, "--D")) a.D_arg = (uint32_t)std::stoul(need("--D"));

        else if (eq(k, "--offsets_npy")) a.offsets_npy = need("--offsets_npy");
        else if (eq(k, "--offsets_u64_bin")) a.offsets_u64_bin = need("--offsets_u64_bin");
        else if (eq(k, "--nlist")) a.nlist_arg = (uint64_t)std::stoull(need("--nlist"));

        else if (eq(k, "--indices_u32_bin")) a.indices_u32_bin = need("--indices_u32_bin");
        else if (eq(k, "--ids_i64_csr_bin")) a.ids_i64_csr_bin = need("--ids_i64_csr_bin");

        else if (eq(k, "--out_dir")) a.out_dir = need("--out_dir");

        else if (eq(k, "--out_sub_dir")) a.out_sub_dir = need("--out_sub_dir");
        else if (eq(k, "--sub_cluster_id_u16")) a.sub_cluster_id_u16 = std::stoi(need("--sub_cluster_id_u16"));
        else if (eq(k, "--emit_tiny_centroid")) a.emit_tiny_centroid = std::stoi(need("--emit_tiny_centroid"));
        else if (eq(k, "--emit_bucketed_vectors")) a.emit_bucketed_vectors = std::stoi(need("--emit_bucketed_vectors"));
        else if (eq(k, "--emit_bucketed_ids")) a.emit_bucketed_ids = std::stoi(need("--emit_bucketed_ids"));

        else if (eq(k, "--threads")) a.threads = std::stoi(need("--threads"));
        else if (eq(k, "--blas_threads")) a.blas_threads = std::stoi(need("--blas_threads"));

        else if (eq(k, "--stage_shm")) a.stage_shm = std::stoi(need("--stage_shm"));
        else if (eq(k, "--shm_dir")) a.shm_dir = need("--shm_dir");

        else if (eq(k, "--cluster_target")) a.cluster_target = std::stoi(need("--cluster_target"));
        else if (eq(k, "--min_bucket_cluster")) a.min_bucket_cluster = std::stoi(need("--min_bucket_cluster"));

        else if (eq(k, "--min_small_bucket_cluster")) a.min_small_bucket_cluster = std::stoi(need("--min_small_bucket_cluster"));
        else if (eq(k, "--small_cluster_target")) a.small_cluster_target = std::stoi(need("--small_cluster_target"));

        else if (eq(k, "--k_cap")) a.k_cap = std::stoi(need("--k_cap"));

        else if (eq(k, "--sample_cap")) a.sample_cap = std::stoi(need("--sample_cap"));
        else if (eq(k, "--kmeans_iter")) a.kmeans_iter = std::stoi(need("--kmeans_iter"));
        else if (eq(k, "--full_refine")) a.full_refine = std::stoi(need("--full_refine"));
        else if (eq(k, "--full_lloyd_iters")) a.full_lloyd_iters = std::stoi(need("--full_lloyd_iters"));
        else if (eq(k, "--reseed_empty_full")) a.reseed_empty_full = std::stoi(need("--reseed_empty_full"));

        else if (eq(k, "--init")) a.init = need("--init");

        else if (eq(k, "--chunk_vecs")) a.chunk_vecs = std::stoi(need("--chunk_vecs"));

        else if (eq(k, "--do_intra_sep")) a.do_intra_sep = std::stoi(need("--do_intra_sep"));
        else if (eq(k, "--do_margin")) a.do_margin = std::stoi(need("--do_margin"));
        else if (eq(k, "--margin_cap")) a.margin_cap = std::stoi(need("--margin_cap"));
        else if (eq(k, "--margin_eps")) a.margin_eps = std::stof(need("--margin_eps"));

        else if (eq(k, "--retry_disaster")) a.retry_disaster = std::stoi(need("--retry_disaster"));
        else if (eq(k, "--retry_reduce_k")) a.retry_reduce_k = std::stoi(need("--retry_reduce_k"));
        else if (eq(k, "--overlap_thr")) a.overlap_thr = std::stof(need("--overlap_thr"));
        else if (eq(k, "--intra_min_thr")) a.intra_min_thr = std::stof(need("--intra_min_thr"));

        else if (eq(k, "--seed")) a.seed = (uint64_t)std::stoull(need("--seed"));

        // ---- GT analysis ----
        else if (eq(k, "--q_npy")) a.q_npy = need("--q_npy");
        else if (eq(k, "--q_f32_bin")) a.q_f32_bin = need("--q_f32_bin");
        else if (eq(k, "--Q")) a.Q_arg = (uint64_t)std::stoull(need("--Q"));
        else if (eq(k, "--Dq")) a.Dq_arg = (uint32_t)std::stoul(need("--Dq"));

        else if (eq(k, "--gt_I_bin")) a.gt_I_bin = need("--gt_I_bin");
        else if (eq(k, "--gt_D_bin")) a.gt_D_bin = need("--gt_D_bin");

        else if (eq(k, "--gt_recompute_q_gt")) a.gt_recompute_q_gt = std::stoi(need("--gt_recompute_q_gt"));
        else if (eq(k, "--gt_dump_cap_per_query")) a.gt_dump_cap_per_query = std::stoi(need("--gt_dump_cap_per_query"));
        else if (eq(k, "--gt_build_pos2bucket")) a.gt_build_pos2bucket = std::stoi(need("--gt_build_pos2bucket"));
        else if (eq(k, "--gt_summary_sample")) a.gt_summary_sample = std::stoi(need("--gt_summary_sample"));
        else if (eq(k, "--gtD_is_l2sqr")) a.gtD_is_l2sqr = std::stoi(need("--gtD_is_l2sqr"));
        else if (eq(k, "--gt_report_both_domains")) a.gt_report_both_domains = std::stoi(need("--gt_report_both_domains"));

        else if (eq(k, "-h") || eq(k, "--help")) { usage(); std::exit(0); }
        else die("unknown arg: " + k);
    }

    // Validate vectors choice
    bool has_vec_npy = !a.vec_npy.empty();
    bool has_vec_bin = !a.vec_f32_bin.empty();
    if (has_vec_npy == has_vec_bin) {
        usage();
        die("must provide exactly one of --vec_npy or --vec_f32_bin");
    }
    if (has_vec_bin) {
        if (a.N_arg == 0 || a.D_arg == 0) die("--vec_f32_bin requires --N and --D");
    }

    // Validate offsets choice
    bool has_off_npy = !a.offsets_npy.empty();
    bool has_off_bin = !a.offsets_u64_bin.empty();
    if (has_off_npy == has_off_bin) {
        usage();
        die("must provide exactly one of --offsets_npy or --offsets_u64_bin");
    }
    if (has_off_bin) {
        if (a.nlist_arg == 0) die("--offsets_u64_bin requires --nlist");
    }

    // Validate ids choice
    bool has_idx_u32 = !a.indices_u32_bin.empty();
    bool has_ids_i64 = !a.ids_i64_csr_bin.empty();
    if (has_idx_u32 == has_ids_i64) {
        usage();
        die("must provide exactly one of --indices_u32_bin or --ids_i64_csr_bin");
    }

    if (a.out_dir.empty()) die("--out_dir required");
    if (a.threads <= 0) die("--threads must be >0");
    if (a.blas_threads <= 0) die("--blas_threads must be >0");
    if (a.cluster_target <= 0) die("--cluster_target must be >0");
    if (a.small_cluster_target <= 0) die("--small_cluster_target must be >0");
    if (a.sample_cap <= 0) die("--sample_cap must be >0");
    if (a.kmeans_iter <= 0) die("--kmeans_iter must be >0");
    if (a.chunk_vecs <= 0) die("--chunk_vecs must be >0");
    if (a.k_cap <= 0) die("--k_cap must be >0");

    if (!(a.init == "kmeanspp" || a.init == "random")) die("--init must be kmeanspp or random");

    bool want_gt = (!a.q_npy.empty() || !a.q_f32_bin.empty() || !a.gt_I_bin.empty() || !a.gt_D_bin.empty());
    if (want_gt) {
        bool has_q_npy = !a.q_npy.empty();
        bool has_q_bin = !a.q_f32_bin.empty();
        if (has_q_npy == has_q_bin) die("[GT] must provide exactly one of --q_npy or --q_f32_bin");
        if (has_q_bin) {
            if (a.Q_arg == 0 || a.Dq_arg == 0) die("[GT] --q_f32_bin requires --Q and --Dq");
        }
        if (a.gt_I_bin.empty()) die("[GT] must provide --gt_I_bin");
        if (a.gt_build_pos2bucket != 0 && a.gt_build_pos2bucket != 1) die("[GT] --gt_build_pos2bucket must be 0/1");
        if (a.gtD_is_l2sqr != 0 && a.gtD_is_l2sqr != 1) die("[GT] --gtD_is_l2sqr must be 0/1");
        if (a.gt_report_both_domains != 0 && a.gt_report_both_domains != 1) die("[GT] --gt_report_both_domains must be 0/1");
    }

    return a;
}

struct BucketStats {
    uint32_t bucket_id = 0;
    uint32_t size = 0;
    uint32_t k = 0;
    float sec = 0.0f;

    float mean_l2 = 0.0f;
    float max_l2 = 0.0f;
    float p50_l2 = 0.0f;
    float p90_l2 = 0.0f;
    float p99_l2 = 0.0f;

    float intra_sep_min = 0.0f;
    float overlap_intra = 0.0f;

    float margin_p50 = 0.0f;
    float ambiguous_frac = 0.0f;

    uint32_t did_retry = 0;
    uint32_t k_used = 0;
};

static void write_bucket_cluster_stats_csv(const std::string& out_dir, const std::vector<BucketStats>& rows) {
    std::ofstream out(out_dir + "/bucket_cluster_stats.csv");
    out << "bucket_id,size,k,k_used,did_retry,sec,mean_l2,max_l2,p50_l2,p90_l2,p99_l2,intra_sep_min,overlap_intra,margin_p50,ambiguous_frac\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.bucket_id << "," << r.size << "," << r.k << "," << r.k_used << "," << r.did_retry << ","
            << r.sec << ","
            << r.mean_l2 << "," << r.max_l2 << "," << r.p50_l2 << "," << r.p90_l2 << "," << r.p99_l2 << ","
            << r.intra_sep_min << "," << r.overlap_intra << ","
            << r.margin_p50 << "," << r.ambiguous_frac << "\n";
    }
}

// -------------------- KMeans initializers on sample --------------------
static void init_centroids_random(
    const float* train, uint32_t ntrain, uint32_t D,
    uint32_t k, std::mt19937_64& rng,
    std::vector<float>& centroids_out) {

    centroids_out.assign((size_t)k * (size_t)D, 0.0f);
    std::vector<uint32_t> idx(ntrain);
    for (uint32_t i = 0; i < ntrain; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    for (uint32_t ci = 0; ci < k; ci++) {
        const float* src = train + (size_t)idx[ci] * (size_t)D;
        std::memcpy(&centroids_out[(size_t)ci * (size_t)D], src, (size_t)D * sizeof(float));
    }
}

static void init_centroids_kmeanspp(
    const float* train, uint32_t ntrain, uint32_t D,
    uint32_t k, std::mt19937_64& rng,
    std::vector<float>& centroids_out) {

    centroids_out.assign((size_t)k * (size_t)D, 0.0f);

    std::uniform_int_distribution<uint32_t> pick0(0, ntrain - 1);
    uint32_t c0 = pick0(rng);
    std::memcpy(&centroids_out[0], train + (size_t)c0 * (size_t)D, (size_t)D * sizeof(float));

    std::vector<float> min_d2(ntrain, std::numeric_limits<float>::max());

    auto update_min_d2_with_center = [&](uint32_t ci) {
        const float* c = &centroids_out[(size_t)ci * (size_t)D];
        for (uint32_t i = 0; i < ntrain; i++) {
            const float* x = train + (size_t)i * (size_t)D;
            float d2 = l2sqr(x, c, D);
            if (d2 < min_d2[i]) min_d2[i] = d2;
        }
    };

    update_min_d2_with_center(0);

    for (uint32_t ci = 1; ci < k; ci++) {
        long double sum = 0.0;
        for (uint32_t i = 0; i < ntrain; i++) sum += (long double)min_d2[i];

        uint32_t chosen = 0;
        if (sum <= 0.0L) {
            std::uniform_int_distribution<uint32_t> pick(0, ntrain - 1);
            chosen = pick(rng);
        } else {
            std::uniform_real_distribution<long double> ur(0.0L, sum);
            long double r = ur(rng);
            long double acc = 0.0L;
            for (uint32_t i = 0; i < ntrain; i++) {
                acc += (long double)min_d2[i];
                if (acc >= r) { chosen = i; break; }
            }
        }

        std::memcpy(&centroids_out[(size_t)ci * (size_t)D],
                    train + (size_t)chosen * (size_t)D,
                    (size_t)D * sizeof(float));

        update_min_d2_with_center(ci);
    }
}

// -------------------- Lloyd kmeans on sample --------------------
static void kmeans_lloyd_train_on_sample(
    const float* train, uint32_t ntrain, uint32_t D,
    uint32_t k, int iters, uint64_t seed,
    const std::string& init_mode,
    std::vector<float>& centroids_out) {

    if (k == 0) k = 1;
    if (k > ntrain) k = ntrain;

    std::mt19937_64 rng(seed);

    if (init_mode == "kmeanspp") init_centroids_kmeanspp(train, ntrain, D, k, rng, centroids_out);
    else init_centroids_random(train, ntrain, D, k, rng, centroids_out);

    std::vector<float> sums((size_t)k * (size_t)D);
    std::vector<uint32_t> counts((size_t)k);

    std::vector<float> dist1((size_t)ntrain);
    std::vector<faiss::idx_t> lab1((size_t)ntrain);

    for (int it = 0; it < iters; it++) {
        faiss::IndexFlatL2 index((int)D);
        index.add((faiss::idx_t)k, centroids_out.data());
        index.search((faiss::idx_t)ntrain, train, 1, dist1.data(), lab1.data());

        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (uint32_t i = 0; i < ntrain; i++) {
            faiss::idx_t cid = lab1[i];
            if (cid < 0 || cid >= (faiss::idx_t)k) cid = 0;
            counts[(size_t)cid]++;

            const float* v = train + (size_t)i * (size_t)D;
            float* s = &sums[(size_t)cid * (size_t)D];
            for (uint32_t d = 0; d < D; d++) s[d] += v[d];
        }

        std::uniform_int_distribution<uint32_t> pick(0, ntrain - 1);
        for (uint32_t ci = 0; ci < k; ci++) {
            uint32_t c = counts[ci];
            float* dst = &centroids_out[(size_t)ci * (size_t)D];
            if (c == 0) {
                uint32_t j = pick(rng);
                const float* src = train + (size_t)j * (size_t)D;
                std::memcpy(dst, src, (size_t)D * sizeof(float));
            } else {
                float inv = 1.0f / (float)c;
                const float* s = &sums[(size_t)ci * (size_t)D];
                for (uint32_t d = 0; d < D; d++) dst[d] = s[d] * inv;
            }
        }
    }
}

static void compute_intra_sep_and_overlap(
    const float* cent, uint32_t k, uint32_t D,
    float p99_l2,
    float* intra_min_out,
    float* overlap_out) {

    float intra_min = 0.0f;
    float overlap = 0.0f;

    if (k >= 2) {
        faiss::IndexFlatL2 c2((int)D);
        c2.add((faiss::idx_t)k, cent);
        std::vector<float> dd((size_t)k * 2);
        std::vector<faiss::idx_t> ii((size_t)k * 2);
        c2.search((faiss::idx_t)k, cent, 2, dd.data(), ii.data());
        intra_min = std::numeric_limits<float>::max();
        for (uint32_t i = 0; i < k; i++) {
            float d = std::sqrt(std::max(0.0f, dd[(size_t)i * 2 + 1]));
            intra_min = std::min(intra_min, d);
        }
        if (intra_min == std::numeric_limits<float>::max()) intra_min = 0.0f;
        if (intra_min > 0.0f) overlap = (2.0f * p99_l2) / intra_min;
        else overlap = std::numeric_limits<float>::infinity();
    }

    *intra_min_out = intra_min;
    *overlap_out = overlap;
}

static void reseed_empty_clusters_full(
    std::vector<float>& cent, uint32_t k, uint32_t D,
    const float* x_bucket, uint32_t sz,
    const std::vector<uint32_t>& counts,
    std::mt19937_64& rng) {

    if (sz == 0) return;
    std::uniform_int_distribution<uint32_t> pick(0, sz - 1);
    for (uint32_t ci = 0; ci < k; ci++) {
        if (counts[ci] != 0) continue;
        uint32_t j = pick(rng);
        std::memcpy(&cent[(size_t)ci * (size_t)D],
                    x_bucket + (size_t)j * (size_t)D,
                    (size_t)D * sizeof(float));
    }
}

// Parse Q,K from your GT filename format:
//   gt_I.int64.Q{Q}.K{K}.bin
//   gt_D.float32.Q{Q}.K{K}.bin
static bool parse_QK_from_gt_filename(const std::string& path, uint64_t* Q_out, uint32_t* K_out) {
    std::string base = std::filesystem::path(path).filename().string();
    std::regex re(R"(.*\.Q([0-9]+)\.K([0-9]+)\.bin$)");
    std::smatch m;
    if (!std::regex_match(base, m, re)) return false;
    if (m.size() != 3) return false;
    *Q_out = (uint64_t)std::stoull(m[1].str());
    *K_out = (uint32_t)std::stoul(m[2].str());
    return true;
}

static void write_gt_summary_txt_base(
    const std::string& out_path,
    uint64_t Q, uint32_t K,
    const std::vector<float>& samp_dgt_l2,
    const std::vector<float>& samp_dcent_l2,
    const std::vector<float>& samp_diff_l2,
    const std::vector<float>& samp_ratio_l2,
    double sec_total,
    uint64_t pairs_done,
    bool had_gtD,
    bool recomputed,
    int gtD_is_l2sqr
) {
    std::ofstream out(out_path);
    out << "GT_ANALYSIS_SUMMARY\n";
    out << "Q=" << Q << " K=" << K << "\n";
    out << "pairs_done=" << pairs_done << "\n";
    out << "had_gtD=" << (had_gtD ? 1 : 0)
        << " recomputed_q_gt=" << (recomputed ? 1 : 0)
        << " gtD_is_l2sqr=" << gtD_is_l2sqr
        << "\n";
    out << "time_sec=" << std::fixed << std::setprecision(3) << sec_total << "\n";
    out << "pairs_per_sec=" << std::fixed << std::setprecision(2)
        << (sec_total > 0 ? (double)pairs_done / sec_total : 0.0) << "\n\n";

    auto dump_quant = [&](const std::string& name, std::vector<float> v) {
        std::vector<float> tmp = std::move(v);
        float p50 = percentile_from_sample(tmp, 0.50f);
        float p90 = percentile_from_sample(tmp, 0.90f);
        float p99 = percentile_from_sample(tmp, 0.99f);
        out << name << "_p50=" << p50 << " p90=" << p90 << " p99=" << p99 << " n=" << tmp.size() << "\n";
    };

    out << "DOMAIN_L2_STATS\n";
    dump_quant("d_q_gt_l2(primary)", samp_dgt_l2);
    dump_quant("d_q_cent_l2", samp_dcent_l2);
    dump_quant("diff_l2(d_gt-d_cent)", samp_diff_l2);
    dump_quant("ratio_l2(d_gt/(d_cent+eps))", samp_ratio_l2);
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_nested(0);
#if defined(_OPENMP) && _OPENMP >= 201307
    omp_set_max_active_levels(1);
#endif
    omp_set_num_threads(a.threads);
#endif

    if (openblas_set_num_threads) openblas_set_num_threads(a.blas_threads);

    // Resolve staging paths
    std::string vec_path_npy = a.vec_npy;
    std::string vec_path_bin = a.vec_f32_bin;
    std::string off_path_npy = a.offsets_npy;
    std::string off_path_bin = a.offsets_u64_bin;
    std::string ids_u32_path = a.indices_u32_bin;
    std::string ids_i64_path = a.ids_i64_csr_bin;

    std::string q_path_npy = a.q_npy;
    std::string q_path_bin = a.q_f32_bin;
    std::string gt_I_path  = a.gt_I_bin;
    std::string gt_D_path  = a.gt_D_bin;

    bool want_gt = (!a.q_npy.empty() || !a.q_f32_bin.empty() || !a.gt_I_bin.empty() || !a.gt_D_bin.empty());

    if (a.stage_shm) {
        std::cerr << "[SHM] staging enabled\n";
        ensure_dir(a.shm_dir + "/in");
        ensure_dir(a.shm_dir + "/out");
        std::cerr << "[SHM] shm_dir=" << a.shm_dir << " free=" << get_free_gib("/dev/shm") << " GiB\n";

        auto stage_one = [&](const std::string& src)->std::string {
            std::string dst = a.shm_dir + "/in/" + std::filesystem::path(src).filename().string();
            copy_file_if_needed(src, dst);
            return dst;
        };

        if (!vec_path_npy.empty()) vec_path_npy = stage_one(vec_path_npy);
        if (!vec_path_bin.empty()) vec_path_bin = stage_one(vec_path_bin);

        if (!off_path_npy.empty()) off_path_npy = stage_one(off_path_npy);
        if (!off_path_bin.empty()) off_path_bin = stage_one(off_path_bin);

        if (!ids_u32_path.empty()) ids_u32_path = stage_one(ids_u32_path);
        if (!ids_i64_path.empty()) ids_i64_path = stage_one(ids_i64_path);

        if (want_gt) {
            if (!q_path_npy.empty()) q_path_npy = stage_one(q_path_npy);
            if (!q_path_bin.empty()) q_path_bin = stage_one(q_path_bin);
            if (!gt_I_path.empty())  gt_I_path = stage_one(gt_I_path);
            if (!gt_D_path.empty())  gt_D_path = stage_one(gt_D_path);
        }

        a.out_dir = a.shm_dir + "/out";
    }
    ensure_dir(a.out_dir);

    std::cerr << "===== CONFIG =====\n";
    if (!vec_path_npy.empty()) std::cerr << "vec_npy      = " << vec_path_npy << "\n";
    if (!vec_path_bin.empty()) std::cerr << "vec_f32_bin  = " << vec_path_bin << " N=" << a.N_arg << " D=" << a.D_arg << "\n";
    if (!off_path_npy.empty()) std::cerr << "offsets_npy  = " << off_path_npy << "\n";
    if (!off_path_bin.empty()) std::cerr << "offsets_bin  = " << off_path_bin << " nlist=" << a.nlist_arg << "\n";
    if (!ids_u32_path.empty()) std::cerr << "indices_u32  = " << ids_u32_path << "\n";
    if (!ids_i64_path.empty()) std::cerr << "ids_i64      = " << ids_i64_path << "\n";
    std::cerr << "out_dir      = " << a.out_dir << "\n";
    std::cerr << "threads      = " << a.threads << " (outer)\n";
    std::cerr << "blas_threads = " << a.blas_threads << "\n";
    std::cerr << "cluster_target= " << a.cluster_target << " min_bucket_cluster=" << a.min_bucket_cluster << "\n";
    std::cerr << "SMALL: min_small_bucket_cluster=" << a.min_small_bucket_cluster
              << " small_cluster_target=" << a.small_cluster_target << "\n";
    std::cerr << "k_cap        = " << a.k_cap << "\n";
    std::cerr << "sample_cap   = " << a.sample_cap << "\n";
    std::cerr << "kmeans_iter(sample)= " << a.kmeans_iter << " init=" << a.init << "\n";
    std::cerr << "full_refine  = " << a.full_refine << " full_lloyd_iters=" << a.full_lloyd_iters
              << " reseed_empty_full=" << a.reseed_empty_full << "\n";
    std::cerr << "chunk_vecs   = " << a.chunk_vecs << "\n";
    std::cerr << "do_intra_sep = " << a.do_intra_sep << " do_margin=" << a.do_margin
              << " margin_cap=" << a.margin_cap << " margin_eps=" << a.margin_eps << "\n";
    std::cerr << "retry_disaster=" << a.retry_disaster << " retry_reduce_k=" << a.retry_reduce_k
              << " overlap_thr=" << a.overlap_thr << " intra_min_thr=" << a.intra_min_thr << "\n";
    if (want_gt) {
        std::cerr << "[GT] enabled\n";
        if (!q_path_npy.empty()) std::cerr << "q_npy        = " << q_path_npy << "\n";
        if (!q_path_bin.empty()) std::cerr << "q_f32_bin    = " << q_path_bin << " Q=" << a.Q_arg << " Dq=" << a.Dq_arg << "\n";
        std::cerr << "gt_I_bin     = " << gt_I_path << "\n";
        if (!gt_D_path.empty()) std::cerr << "gt_D_bin     = " << gt_D_path << "\n";
        std::cerr << "gt_recompute_q_gt=" << a.gt_recompute_q_gt
                  << " gt_dump_cap_per_query=" << a.gt_dump_cap_per_query
                  << " gt_build_pos2bucket=" << a.gt_build_pos2bucket
                  << " gt_summary_sample=" << a.gt_summary_sample
                  << " gtD_is_l2sqr=" << a.gtD_is_l2sqr
                  << " gt_report_both_domains=" << a.gt_report_both_domains
                  << "\n";
    }
    std::cerr << "==================\n";

    // --------- Load vectors xb ---------
    MMap mm_vec;
    const float* xb = nullptr;
    uint64_t N = 0;
    uint32_t D = 0;

    if (!vec_path_npy.empty()) {
        NpyArrayView vec = load_npy_view(mm_vec, vec_path_npy);
        if (vec.descr != "<f4") die("vec_npy dtype mismatch: got " + vec.descr + " want <f4");
        if (vec.shape.size() != 2) die("vec_npy must be 2D [N,D]");
        N = vec.shape[0];
        D = (uint32_t)vec.shape[1];
        xb = (const float*)vec.data;
    } else {
        N = a.N_arg;
        D = a.D_arg;
        uint64_t bytes = file_size(vec_path_bin);
        uint64_t expect = N * (uint64_t)D * sizeof(float);
        if (bytes != expect) die("vec_f32_bin size mismatch: got=" + std::to_string(bytes) + " expect=" + std::to_string(expect));
        mm_vec.open_ro(vec_path_bin);
        xb = (const float*)mm_vec.addr;
    }

    // --------- Load offsets ---------
    const uint64_t* offsets = nullptr;
    uint64_t nlist = 0;

    MMap mm_off;
    if (!off_path_npy.empty()) {
        NpyArrayView off = load_npy_view(mm_off, off_path_npy);
        if (!(off.descr == "<u8" || off.descr == "<i8")) die("offsets_npy dtype must be <u8/<i8");
        uint64_t off_len = 0;
        if (off.shape.size() == 1) off_len = off.shape[0];
        else if (off.shape.size() == 2) {
            if (off.shape[1] != 1) die("offsets 2D must be (n,1)");
            off_len = off.shape[0];
        } else die("offsets must be 1D or 2D");
        if (off_len < 2) die("offsets too small");
        nlist = off_len - 1;
        offsets = (const uint64_t*)off.data;
    } else {
        nlist = a.nlist_arg;
        uint64_t bytes = file_size(off_path_bin);
        uint64_t expect = (nlist + 1) * sizeof(uint64_t);
        if (bytes != expect) die("offsets_u64_bin size mismatch: got=" + std::to_string(bytes) + " expect=" + std::to_string(expect));
        mm_off.open_ro(off_path_bin);
        offsets = (const uint64_t*)mm_off.addr;
    }

    uint64_t total_in_csr = offsets[nlist];
    if (total_in_csr != N) die("CSR total mismatch: offsets[nlist] != N (offsets[nlist]=" + std::to_string(total_in_csr) + " N=" + std::to_string(N) + ")");

    // --------- Load ids mapping (either u32 or i64) ---------
    MMap mm_ids;
    const uint32_t* indices_u32 = nullptr;
    const int64_t* ids_i64 = nullptr;
    bool use_u32 = !ids_u32_path.empty();

    if (use_u32) {
        uint64_t bytes = file_size(ids_u32_path);
        uint64_t expect = N * sizeof(uint32_t);
        if (bytes != expect) die("indices_u32_bin size mismatch: got=" + std::to_string(bytes) + " expect=" + std::to_string(expect));
        mm_ids.open_ro(ids_u32_path);
        indices_u32 = (const uint32_t*)mm_ids.addr;
    } else {
        uint64_t bytes = file_size(ids_i64_path);
        uint64_t expect = N * sizeof(int64_t);
        if (bytes != expect) die("ids_i64_csr_bin size mismatch: got=" + std::to_string(bytes) + " expect=" + std::to_string(expect));
        mm_ids.open_ro(ids_i64_path);
        ids_i64 = (const int64_t*)mm_ids.addr;
    }

    std::cerr << "Loaded: N=" << N << " D=" << D << " nlist=" << nlist << " total_in_csr=" << total_in_csr << "\n";

    // --------- bucket sizes ---------
    std::vector<uint32_t> sizes_u32(nlist);
    uint64_t non_empty=0, sum_sizes=0, max_size=0, min_nonempty=std::numeric_limits<uint64_t>::max();
    for (uint64_t b=0;b<nlist;b++){
        uint64_t sz = offsets[b+1]-offsets[b];
        sum_sizes += sz;
        if (sz>0){ non_empty++; min_nonempty = std::min(min_nonempty, sz); max_size = std::max(max_size, sz); }
        sizes_u32[b] = (uint32_t)sz;
    }

    auto percentile_bucket = [&](double q)->uint64_t{
        std::vector<uint32_t> tmp; tmp.reserve(nlist);
        for (uint64_t b=0;b<nlist;b++) tmp.push_back(sizes_u32[b]);
        size_t k = (size_t)std::floor(q*(tmp.size()-1));
        std::nth_element(tmp.begin(), tmp.begin()+k, tmp.end());
        return (uint64_t)tmp[k];
    };

    std::cerr << "==== Bucket size stats ====\n";
    std::cerr << "nlist          = " << nlist << "\n";
    std::cerr << "non_empty      = " << non_empty << " (" << (100.0*double(non_empty)/double(nlist)) << "%)\n";
    std::cerr << "sum_sizes      = " << sum_sizes << "\n";
    std::cerr << "min_nonempty   = " << (non_empty?min_nonempty:0) << "\n";
    std::cerr << "max            = " << max_size << "\n";
    std::cerr << "mean_all       = " << (double)sum_sizes / (double)nlist << "\n";
    std::cerr << "mean_nonempty  = " << (non_empty? (double)sum_sizes / (double)non_empty : 0.0) << "\n";
    std::cerr << "p50/p90/p99    = " << percentile_bucket(0.50) << " / " << percentile_bucket(0.90) << " / " << percentile_bucket(0.99) << "\n";

    // --------- Optional reorder by bucket -> bucketed vectors/ids ---------
    // NOTE: CSR already stores ids in bucket order. bucketed_ids.u32.bin is just a u32 view.
    // bucketed_vectors.f32.bin is large (N*D float32) and can be skipped to save shm.
    std::string out_bucketed_vec = a.out_dir + "/bucketed_vectors.f32.bin";
    std::string out_bucketed_ids = a.out_dir + "/bucketed_ids.u32.bin";

    auto get_id_u32_at_csrpos = [&](uint64_t csr_pos)->uint32_t {
        uint64_t id64 = 0;
        if (use_u32) id64 = (uint64_t)indices_u32[csr_pos];
        else {
            int64_t v = ids_i64[csr_pos];
            if (v < 0) die("ids_i64 contains negative id at csr_pos=" + std::to_string(csr_pos));
            id64 = (uint64_t)v;
        }
        if (id64 >= N) die("id out of range: id=" + std::to_string(id64) + " N=" + std::to_string(N));
        if (id64 > std::numeric_limits<uint32_t>::max()) die("id exceeds u32 range: id=" + std::to_string(id64));
        return (uint32_t)id64;
    };

    // Emit reorder files as requested (can be skipped entirely)
    const bool want_bucketed_vec = (a.emit_bucketed_vectors != 0);
    const bool want_bucketed_ids = (a.emit_bucketed_ids != 0);

    bool have_vec = std::filesystem::exists(out_bucketed_vec);
    bool have_ids = std::filesystem::exists(out_bucketed_ids);

    if (want_bucketed_vec && !have_vec) {
        uint64_t bytes_vec = N*(uint64_t)D*sizeof(float);
        std::cerr << "---- Reordering by bucket (vectors) -> " << out_bucketed_vec
                  << " (" << (double)bytes_vec/(1024.0*1024.0*1024.0) << " GiB)\n";
        double t0 = now_sec();
        MMap mm_out_vec;
        mm_out_vec.open_rw_create(out_bucketed_vec, (size_t)bytes_vec);
        float* outv = (float*)mm_out_vec.addr;

        #pragma omp parallel
        {
            std::vector<float> buf((size_t)a.chunk_vecs*(size_t)D);

            #pragma omp for schedule(dynamic, 256)
            for (int64_t bi = 0; bi < (int64_t)nlist; bi++) {
                uint64_t b = (uint64_t)bi;
                uint64_t beg = offsets[b], end = offsets[b+1];
                uint64_t sz = end - beg;
                uint64_t i = 0;
                while (i < sz) {
                    uint64_t take = std::min<uint64_t>((uint64_t)a.chunk_vecs, sz - i);
                    for (uint64_t j = 0; j < take; j++) {
                        uint64_t csr_pos = beg + i + j;
                        uint32_t id = get_id_u32_at_csrpos(csr_pos);
                        std::memcpy(&buf[(size_t)j*(size_t)D], xb + (uint64_t)id*(uint64_t)D, (size_t)D*sizeof(float));
                    }
                    std::memcpy(outv + (beg + i)*(uint64_t)D, buf.data(), (size_t)take*(size_t)D*sizeof(float));
                    i += take;
                }
            }
        }
        msync(mm_out_vec.addr, mm_out_vec.len, MS_SYNC);
        double sec = elapsed_sec(t0);
        double gib = (double)bytes_vec/(1024.0*1024.0*1024.0);
        std::cerr << "Reorder vectors done in " << sec << " sec | write=" << gib << " GiB | throughput=" << (gib/sec) << " GiB/s\n";
        have_vec = true;
    } else if (want_bucketed_vec) {
        std::cerr << "[INFO] bucketed_vectors.f32.bin exists; skip reorder vectors.\n";
    }

    if (want_bucketed_ids && !have_ids) {
        uint64_t bytes_ids = N*sizeof(uint32_t);
        std::cerr << "---- Emitting bucketed ids -> " << out_bucketed_ids
                  << " (" << (double)bytes_ids/(1024.0*1024.0*1024.0) << " GiB)\n";
        double t0 = now_sec();
        MMap mm_out_ids;
        mm_out_ids.open_rw_create(out_bucketed_ids, (size_t)bytes_ids);
        uint32_t* outid = (uint32_t*)mm_out_ids.addr;

        #pragma omp parallel for schedule(dynamic, 2048)
        for (int64_t p = 0; p < (int64_t)N; p++) {
            outid[(uint64_t)p] = get_id_u32_at_csrpos((uint64_t)p);
        }

        msync(mm_out_ids.addr, mm_out_ids.len, MS_SYNC);
        double sec = elapsed_sec(t0);
        double gib = (double)bytes_ids/(1024.0*1024.0*1024.0);
        std::cerr << "Emit ids done in " << sec << " sec | write=" << gib << " GiB | throughput=" << (gib/sec) << " GiB/s\n";
        have_ids = true;
    } else if (want_bucketed_ids) {
        std::cerr << "[INFO] bucketed_ids.u32.bin exists; skip emit ids.\n";
    }

    // mmap bucketed vectors (optional)
    MMap mm_bvec;
    const float* bvec = nullptr;
    if (want_bucketed_vec) {
        if (!have_vec) die("emit_bucketed_vectors=1 but bucketed_vectors.f32.bin is missing");
        mm_bvec.open_ro(out_bucketed_vec);
        bvec = (const float*)mm_bvec.addr;
        if (mm_bvec.len != (size_t)(N*(uint64_t)D*sizeof(float))) die("bucketed_vectors size mismatch");
    }

    // mmap bucketed ids (optional)
    MMap mm_bids;
    const uint32_t* bids = nullptr;
    if (want_bucketed_ids) {
        if (!have_ids) die("emit_bucketed_ids=1 but bucketed_ids.u32.bin is missing");
        mm_bids.open_ro(out_bucketed_ids);
        bids = (const uint32_t*)mm_bids.addr;
        if (mm_bids.len != (size_t)(N*sizeof(uint32_t))) die("bucketed_ids size mismatch");
    }

    // --------- decide k per bucket (LARGE + SMALL) ---------
    std::vector<uint32_t> k_per_bucket(nlist, 0);
    uint64_t totalK = 0;

    uint64_t cnt_large=0, cnt_small=0;
    uint64_t cnt_tiny=0;
    for (uint64_t b=0;b<nlist;b++){
        uint32_t sz = sizes_u32[b];
        uint64_t k = 0;

        if (sz == 0) {
            k = 0;
        } else if ((int)sz < a.min_small_bucket_cluster) {
            // Keep structure complete: tiny (unclustered) buckets are treated as a single cluster (k=1).
            k = 1;
            cnt_tiny++;
        } else if ((int)sz >= a.min_bucket_cluster) {
            k = ceil_div_u64(sz, (uint64_t)a.cluster_target);
            cnt_large++;
        } else {
            // SMALL: min_small_bucket_cluster <= sz < min_bucket_cluster
            k = ceil_div_u64(sz, (uint64_t)a.small_cluster_target);
            cnt_small++;
        }

        if (k > 0) {
            if (k > (uint64_t)a.k_cap) k = (uint64_t)a.k_cap;
            k_per_bucket[b] = (uint32_t)k;
            totalK += k;
        }
    }
    std::cerr << "Buckets clustered: large=" << cnt_large << " small=" << cnt_small << " tiny=" << cnt_tiny << "\n";

    std::cerr << "Total clusters (capped) = " << totalK << "\n";

    // --------- cluster_offsets ---------
    std::string out_cluster_offsets = a.out_dir + "/cluster_offsets.u64.bin";
    {
        std::ofstream f(out_cluster_offsets, std::ios::binary|std::ios::trunc);
        std::vector<uint64_t> tmp(nlist+1, 0);
        uint64_t acc=0;
        for (uint64_t b=0;b<nlist;b++){ tmp[b]=acc; acc += k_per_bucket[b]; }
        tmp[nlist]=acc;
        f.write((const char*)tmp.data(), (std::streamsize)tmp.size()*sizeof(uint64_t));
        f.flush();
        if (acc != totalK) die("totalK mismatch");
    }
    MMap mm_coff;
    mm_coff.open_ro(out_cluster_offsets);
    const uint64_t* cluster_offsets = (const uint64_t*)mm_coff.addr;

    // --------- outputs ---------
    std::string out_centroids   = a.out_dir + "/centroids.f32.bin";
    std::string out_cluster_ids = a.out_dir + "/cluster_ids.u32.bin";
    std::string out_assignments = a.out_dir + "/assignments_global.u32.bin";

    MMap mm_cent_out, mm_cids_out, mm_asg_out;
    mm_cent_out.open_rw_create(out_centroids, (size_t)(totalK*(uint64_t)D*sizeof(float)));
    mm_cids_out.open_rw_create(out_cluster_ids, (size_t)(totalK*sizeof(uint32_t)));
    mm_asg_out.open_rw_create(out_assignments, (size_t)(N*sizeof(uint32_t)));

    float* cent_out = (float*)mm_cent_out.addr;
    uint32_t* cids_out = (uint32_t*)mm_cids_out.addr;
    uint32_t* assignments = (uint32_t*)mm_asg_out.addr;
    std::fill(assignments, assignments+N, std::numeric_limits<uint32_t>::max());

    // --------- clustering ---------
    std::cerr << "---- Clustering buckets (large>= " << a.min_bucket_cluster
              << " OR small>= " << a.min_small_bucket_cluster << ") ...\n";
    std::vector<BucketStats> rows;
    rows.reserve(200000);
    std::mutex rows_mu;

    std::atomic<uint64_t> vecs_done{0};
    double t0 = now_sec();

#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t bi = 0; bi < (int64_t)nlist; bi++) {
        uint64_t b = (uint64_t)bi;
        uint32_t sz = sizes_u32[b];
        uint32_t k0 = k_per_bucket[b];
        if (k0 == 0) continue;

        double tb = now_sec();
        uint64_t beg = offsets[b];

        // Access bucket vectors:
        //  - If bucketed_vectors was emitted, use contiguous bvec.
        //  - Else, gather xb[id] into a thread-local buffer for this bucket.
        const float* x = nullptr;
        static thread_local std::vector<float> tls_xbuf;
        if (bvec) {
            x = bvec + beg*(uint64_t)D;
        } else {
            tls_xbuf.resize((size_t)sz * (size_t)D);
            float* dst = tls_xbuf.data();
            for (uint32_t i2 = 0; i2 < sz; i2++) {
                uint64_t csr_pos = beg + (uint64_t)i2;
                uint32_t id = get_id_u32_at_csrpos(csr_pos);
                std::memcpy(dst + (size_t)i2*(size_t)D, xb + (uint64_t)id*(uint64_t)D, (size_t)D*sizeof(float));
            }
            x = dst;
        }


        // FAST PATH: tiny buckets (sz < min_small_bucket_cluster) are treated as k=1 without k-means
        if (sz > 0 && (int)sz < a.min_small_bucket_cluster) {
            // k0 should be 1 for tiny buckets
            uint64_t coff = cluster_offsets[b];
            // Assign all points to local cluster 0
            for (uint32_t i2 = 0; i2 < sz; i2++) assignments[beg + (uint64_t)i2] = 0;
            // Emit centroid (mean) if requested; else zeros
            float* centp = cent_out + coff*(uint64_t)D;
            if (a.emit_tiny_centroid) {
                std::fill(centp, centp + D, 0.0f);
                for (uint32_t i2 = 0; i2 < sz; i2++) {
                    const float* v = x + (uint64_t)i2*(uint64_t)D;
                    for (uint32_t d = 0; d < D; d++) centp[d] += v[d];
                }
                float inv = 1.0f / (float)sz;
                for (uint32_t d = 0; d < D; d++) centp[d] *= inv;
            } else {
                std::fill(centp, centp + D, 0.0f);
            }
            // cluster id -> parent bucket id
            cids_out[coff] = (uint32_t)b;

            BucketStats r;
            r.bucket_id = (uint32_t)b;
            r.size = sz;
            r.k = k0;
            r.k_used = 1;
            r.did_retry = false;
            r.sec = (float)elapsed_sec(tb);
            r.mean_l2 = 0.0f;
            r.max_l2  = 0.0f;
            r.p50_l2  = 0.0f;
            r.p90_l2  = 0.0f;
            r.p99_l2  = 0.0f;
            r.intra_sep_min = 0.0f;
            r.overlap_intra = 0.0f;
            r.margin_p50 = 0.0f;
            r.ambiguous_frac = 0.0f;
            { std::lock_guard<std::mutex> lk(rows_mu); rows.push_back(r); }
            vecs_done.fetch_add(sz, std::memory_order_relaxed);
            continue;
        }

        auto run_one = [&](uint32_t k, uint64_t seed,
                           float* mean_l2_out, float* max_l2_out,
                           float* p50_out, float* p90_out, float* p99_out,
                           float* intra_min_out, float* overlap_out,
                           float* margin_p50_out, float* ambiguous_frac_out,
                           std::vector<float>& cent_final_out,
                           uint32_t* k_used_out) {

            uint32_t ntrain = (uint32_t)std::min<uint64_t>((uint64_t)a.sample_cap, (uint64_t)sz);
            std::vector<float> train((size_t)ntrain*(size_t)D);
            {
                std::mt19937_64 rng(seed + b*1315423911ULL);
                if (ntrain == sz) {
                    std::memcpy(train.data(), x, (size_t)sz*(size_t)D*sizeof(float));
                } else {
                    std::uniform_int_distribution<uint64_t> dist(0, (uint64_t)sz-1);
                    for (uint32_t i=0;i<ntrain;i++){
                        uint64_t j = dist(rng);
                        std::memcpy(&train[(size_t)i*(size_t)D], x + j*(uint64_t)D, (size_t)D*sizeof(float));
                    }
                }
            }

            uint32_t k_eff = k;
            if (k_eff > ntrain) k_eff = ntrain;
            if (k_eff == 0) k_eff = 1;

            std::vector<float> cent;
            kmeans_lloyd_train_on_sample(
                train.data(), ntrain, D,
                k_eff, a.kmeans_iter, seed ^ (b*2654435761ULL),
                a.init,
                cent
            );

            // If k_eff < requested k, replicate last centroid to fill.
            if (k_eff < k) {
                cent.resize((size_t)k * (size_t)D);
                for (uint32_t ci = k_eff; ci < k; ci++) {
                    std::memcpy(&cent[(size_t)ci*(size_t)D],
                                &cent[(size_t)(k_eff-1)*(size_t)D],
                                (size_t)D*sizeof(float));
                }
                k_eff = k;
            }

            faiss::IndexFlatL2 cindex((int)D);
            cindex.add((faiss::idx_t)k, cent.data());

            std::vector<float> sums((size_t)k*(size_t)D);
            std::vector<uint32_t> counts((size_t)k);
            std::vector<float> dist1((size_t)a.chunk_vecs);
            std::vector<faiss::idx_t> lab1((size_t)a.chunk_vecs);

            Reservoir dist_res(65536, seed + b*97531ULL);
            Reservoir margin_res((size_t)a.margin_cap, seed + b*424242ULL);

            uint64_t ambiguous_cnt=0, margin_seen=0;
            float max_l2=0.0f;
            double sum_l2=0.0;

            std::mt19937_64 full_rng(seed + b*99991ULL);

            auto one_full_pass = [&](bool collect) {
                std::fill(sums.begin(), sums.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                uint64_t pos=0;
                while(pos<sz){
                    uint64_t take = std::min<uint64_t>((uint64_t)a.chunk_vecs, (uint64_t)sz-pos);
                    const float* xp = x + pos*(uint64_t)D;

                    cindex.search((faiss::idx_t)take, xp, 1, dist1.data(), lab1.data());

                    for(uint64_t i=0;i<take;i++){
                        faiss::idx_t cid = lab1[(size_t)i];
                        if (cid<0 || cid>=(faiss::idx_t)k) cid=0;
                        counts[(size_t)cid]++;

                        const float* v = xp + i*(uint64_t)D;
                        float* s = &sums[(size_t)cid*(size_t)D];
                        for(uint32_t d=0; d<D; d++) s[d] += v[d];

                        assignments[beg + pos + i] = (uint32_t)cid;

                        if (collect) {
                            float l2sq = dist1[(size_t)i];
                            float l2v = std::sqrt(std::max(0.0f,l2sq));
                            sum_l2 += l2v;
                            max_l2 = std::max(max_l2, l2v);
                            dist_res.push(l2v);

                            if (a.do_margin && a.margin_cap>0) {
                                // Margin sampling (approx):
                                bool do_this = false;
                                if (margin_seen < (uint64_t)a.margin_cap) do_this = true;
                                else {
                                    std::uniform_int_distribution<uint64_t> dist(0, margin_seen);
                                    uint64_t j = dist(margin_res.rng);
                                    if (j < (uint64_t)a.margin_cap) do_this = true;
                                }
                                if (do_this) {
                                    float d2[2]; faiss::idx_t id2[2];
                                    cindex.search(1, v, 2, d2, id2);
                                    float d1 = std::sqrt(std::max(0.0f, d2[0]));
                                    float dB = std::sqrt(std::max(0.0f, d2[1]));
                                    float m = dB - d1;
                                    margin_res.push(m);
                                    if (m < a.margin_eps) ambiguous_cnt++;
                                }
                                margin_seen++;
                            }
                        }
                    }
                    pos += take;
                }

                if (a.reseed_empty_full) {
                    reseed_empty_clusters_full(cent, k, D, x, sz, counts, full_rng);
                }

                for(uint32_t ci=0; ci<k; ci++){
                    uint32_t c = counts[ci];
                    float* dst = &cent[(size_t)ci*(size_t)D];
                    if (!c) continue;
                    float inv = 1.0f/(float)c;
                    const float* s = &sums[(size_t)ci*(size_t)D];
                    for(uint32_t d=0; d<D; d++) dst[d] = s[d]*inv;
                }

                cindex.reset();
                cindex.add((faiss::idx_t)k, cent.data());
            };

            if (a.full_refine) {
                int passes = std::max(1, a.full_lloyd_iters);
                for (int pass=0; pass<passes; pass++) one_full_pass(pass==passes-1);
            } else {
                one_full_pass(true);
            }

            float mean_l2 = (sz>0) ? (float)(sum_l2/(double)sz) : 0.0f;
            std::vector<float> dist_samp = dist_res.buf;
            float p50 = percentile_from_sample(dist_samp, 0.50f);
            float p90 = percentile_from_sample(dist_samp, 0.90f);
            float p99 = percentile_from_sample(dist_samp, 0.99f);

            float intra_min=0.0f, overlap=0.0f;
            if (a.do_intra_sep) compute_intra_sep_and_overlap(cent.data(), k, D, p99, &intra_min, &overlap);

            float margin_p50=0.0f, ambiguous_frac=0.0f;
            if (a.do_margin && !margin_res.buf.empty()) {
                std::vector<float> ms = margin_res.buf;
                margin_p50 = percentile_from_sample(ms, 0.50f);
                ambiguous_frac = (margin_seen>0) ? (float)ambiguous_cnt/(float)margin_seen : 0.0f;
            }

            *mean_l2_out = mean_l2;
            *max_l2_out  = max_l2;
            *p50_out = p50;
            *p90_out = p90;
            *p99_out = p99;
            *intra_min_out = intra_min;
            *overlap_out = overlap;
            *margin_p50_out = margin_p50;
            *ambiguous_frac_out = ambiguous_frac;

            cent_final_out = std::move(cent);
            *k_used_out = k;
        };

        uint32_t k = k0;
        uint32_t k_used = k0;
        uint32_t did_retry = 0;

        float mean_l2=0, max_l2=0, p50=0, p90=0, p99=0, intra_min=0, overlap=0, margin_p50=0, ambiguous_frac=0;
        std::vector<float> cent;

        uint64_t seed1 = a.seed ^ (b*11400714819323198485ULL);
        run_one(k, seed1, &mean_l2, &max_l2, &p50, &p90, &p99, &intra_min, &overlap, &margin_p50, &ambiguous_frac, cent, &k_used);

        if (a.retry_disaster) {
            bool bad = false;
            if (a.do_intra_sep) {
                if (intra_min > 0.0f && intra_min < a.intra_min_thr) bad = true;
                if (std::isfinite(overlap) && overlap > a.overlap_thr) bad = true;
                if (!std::isfinite(overlap)) bad = true;
            }
            if (bad) {
                did_retry = 1;
                uint32_t k2 = k;
                if (a.retry_reduce_k) k2 = std::max<uint32_t>(1u, k / 2u);

                float mean2=0, max2=0, p50_2=0, p90_2=0, p99_2=0, intra2=0, overlap2=0, mp50_2=0, amb2=0;
                std::vector<float> cent2;
                uint32_t k_used2 = k2;

                uint64_t seed2 = seed1 + 0x9e3779b97f4a7c15ULL;
                run_one(k2, seed2, &mean2, &max2, &p50_2, &p90_2, &p99_2, &intra2, &overlap2, &mp50_2, &amb2, cent2, &k_used2);

                bool take_retry = false;
                if (!std::isfinite(overlap)) take_retry = true;
                else if (std::isfinite(overlap2) && overlap2 < overlap) take_retry = true;

                if (take_retry) {
                    k = k2;
                    k_used = k_used2;
                    mean_l2=mean2; max_l2=max2; p50=p50_2; p90=p90_2; p99=p99_2;
                    intra_min=intra2; overlap=overlap2; margin_p50=mp50_2; ambiguous_frac=amb2;
                    cent.swap(cent2);
                }
            }
        }

        // Write outputs (disjoint ranges across buckets)
        uint64_t coff = cluster_offsets[b];

        // Ensure we write exactly k0 centroids per bucket (even if retry used smaller k)
        if (k < k0) {
            cent.resize((size_t)k0 * (size_t)D);
            for (uint32_t ci = k; ci < k0; ci++) {
                std::memcpy(&cent[(size_t)ci*(size_t)D],
                            &cent[(size_t)(k-1)*(size_t)D],
                            (size_t)D*sizeof(float));
            }
        } else if (k > k0) {
            cent.resize((size_t)k0 * (size_t)D);
        }

        std::memcpy(cent_out + coff*(uint64_t)D, cent.data(), (size_t)k0*(size_t)D*sizeof(float));
        for(uint32_t ci=0; ci<k0; ci++) cids_out[coff + ci] = (uint32_t)b;

        BucketStats r;
        r.bucket_id = (uint32_t)b;
        r.size = sz;
        r.k = k0;
        r.k_used = k;
        r.did_retry = did_retry;
        r.sec = (float)elapsed_sec(tb);
        r.mean_l2 = mean_l2;
        r.max_l2  = max_l2;
        r.p50_l2  = p50;
        r.p90_l2  = p90;
        r.p99_l2  = p99;
        r.intra_sep_min = intra_min;
        r.overlap_intra = overlap;
        r.margin_p50 = margin_p50;
        r.ambiguous_frac = ambiguous_frac;

        {
            std::lock_guard<std::mutex> lk(rows_mu);
            rows.push_back(r);
        }

        vecs_done.fetch_add(sz, std::memory_order_relaxed);
    }

    double sec = elapsed_sec(t0);
    double mvec_s = (double)vecs_done.load()/1e6/sec;
    std::cerr << "Clustering done in " << sec << " sec | approx=" << mvec_s << " Mvec/s\n";

    std::sort(rows.begin(), rows.end(), [](const BucketStats& a, const BucketStats& b){ return a.bucket_id < b.bucket_id; });
    write_bucket_cluster_stats_csv(a.out_dir, rows);

    msync(mm_cent_out.addr, mm_cent_out.len, MS_SYNC);
    msync(mm_cids_out.addr, mm_cids_out.len, MS_SYNC);
    msync(mm_asg_out.addr, mm_asg_out.len, MS_SYNC);

    // -------------------- SUB-CSR (bucket -> in-bucket clusters) (optional) --------------------
    // Goal: write a CSR in the SAME spirit as input CSR, but where the grouping key is:
    //   sub-bucket = (bucket_id, local_cluster_id)
    //
    // Outputs under out_sub_dir (single directory):
    //   sub_offsets.u64.bin          [n_sub+1] uint64
    //   sub_ids.i64.bin              [N]       int64   (base ids in sub-bucket order)
    //   sub_bucket_id.u32.bin        [n_sub]   uint32  (parent bucket for each sub-bucket)
    //   sub_cluster_id.(u16|u32).bin [n_sub]   uint16/uint32 (local cluster id within bucket)
    //   sub_centroids.f32.bin        [n_sub, D] float32 (centroid per sub-bucket; for unclustered buckets we store mean)
    //   bucket_sub_offsets.u64.bin   [nlist+1] uint64  (bucket -> range of sub-buckets)
    //
    // Two-pass method:
    //   Pass1: compute per-sub-bucket counts + (optional) centroids, then prefix-sum -> sub_offsets.
    //   Pass2: fill sub_ids by stable bucket scan + per-cluster write pointers.

    if (!a.out_sub_dir.empty()) {
        ensure_dir(a.out_sub_dir);

        const std::string p_sub_offsets = a.out_sub_dir + "/sub_offsets.u64.bin";
        const std::string p_sub_ids     = a.out_sub_dir + "/sub_ids.i64.bin";
        const std::string p_sub_bid     = a.out_sub_dir + "/sub_bucket_id.u32.bin";
        const std::string p_sub_cid16   = a.out_sub_dir + "/sub_cluster_id.u16.bin";
        const std::string p_sub_cid32   = a.out_sub_dir + "/sub_cluster_id.u32.bin";
        const std::string p_sub_cent    = a.out_sub_dir + "/sub_centroids.f32.bin";
        const std::string p_bsub_off    = a.out_sub_dir + "/bucket_sub_offsets.u64.bin";

        std::cerr << "---- Building sub-CSR under: " << a.out_sub_dir << "\n";

        // bucket_sub_offsets: bucket -> contiguous range of sub-buckets
        std::vector<uint64_t> bucket_sub_offsets(nlist + 1, 0);
        uint64_t n_sub = 0;
        for (uint64_t b = 0; b < nlist; b++) {
            bucket_sub_offsets[b] = n_sub;
            uint64_t kb = (uint64_t)k_per_bucket[b];
            if (kb == 0) kb = 1; // unclustered bucket => single sub-bucket
            n_sub += kb;
        }
        bucket_sub_offsets[nlist] = n_sub;

        std::cerr << "sub-buckets n_sub = " << n_sub << " (avg per non-empty bucket ~ "
                  << (non_empty ? (double)n_sub / (double)non_empty : 0.0) << ")\n";

        // meta + sizes
        std::vector<uint32_t> sub_bucket_id((size_t)n_sub, 0);
        std::vector<uint64_t> sub_sizes((size_t)n_sub, 0);

        // cluster id storage (either u16 or u32)
        std::vector<uint16_t> sub_cluster_id_u16;
        std::vector<uint32_t> sub_cluster_id_u32;
        if (a.sub_cluster_id_u16) sub_cluster_id_u16.assign((size_t)n_sub, 0);
        else sub_cluster_id_u32.assign((size_t)n_sub, 0);

        // centroids per sub-bucket
        std::vector<float> sub_centroids((size_t)n_sub * (size_t)D, 0.0f);

        double t_sub0 = now_sec();

#pragma omp parallel for schedule(dynamic, 256)
        for (int64_t bi = 0; bi < (int64_t)nlist; bi++) {
            uint64_t b = (uint64_t)bi;
            uint64_t beg = offsets[b], end = offsets[b+1];
            uint64_t sz = end - beg;

            uint64_t sub0 = bucket_sub_offsets[b];
            uint32_t kb0 = k_per_bucket[b];

            if (kb0 == 0) {
                // single sub-bucket for this bucket
                sub_bucket_id[(size_t)sub0] = (uint32_t)b;
                if (a.sub_cluster_id_u16) sub_cluster_id_u16[(size_t)sub0] = 0;
                else sub_cluster_id_u32[(size_t)sub0] = 0;
                sub_sizes[(size_t)sub0] = sz;

                // centroid for tiny/unclustered bucket:
                //   emit_tiny_centroid=1 -> mean of bucket vectors (exact)
                //   emit_tiny_centroid=0 -> keep zeros (no extra compute)
                if (a.emit_tiny_centroid && sz > 0) {
                    std::vector<double> acc((size_t)D, 0.0);
                    if (bvec) {
                        const float* x = bvec + beg * (uint64_t)D;
                        for (uint64_t i = 0; i < sz; i++) {
                            const float* v = x + i * (uint64_t)D;
                            for (uint32_t d = 0; d < D; d++) acc[d] += (double)v[d];
                        }
                    } else {
                        for (uint64_t i = 0; i < sz; i++) {
                            uint64_t csr_pos = beg + i;
                            uint32_t id = get_id_u32_at_csrpos(csr_pos);
                            const float* v = xb + (uint64_t)id * (uint64_t)D;
                            for (uint32_t d = 0; d < D; d++) acc[d] += (double)v[d];
                        }
                    }
                    double inv = 1.0 / (double)sz;
                    float* c = &sub_centroids[(size_t)sub0 * (size_t)D];
                    for (uint32_t d = 0; d < D; d++) c[d] = (float)(acc[d] * inv);
                }
            } else {
                // clustered bucket: kb0 sub-buckets
                for (uint32_t c = 0; c < kb0; c++) {
                    uint64_t s = sub0 + (uint64_t)c;
                    sub_bucket_id[(size_t)s] = (uint32_t)b;
                    if (a.sub_cluster_id_u16) sub_cluster_id_u16[(size_t)s] = (uint16_t)c;
                    else sub_cluster_id_u32[(size_t)s] = (uint32_t)c;
                }

                // sizes by scanning assignments in this bucket
                std::vector<uint32_t> cnt((size_t)kb0, 0);
                for (uint64_t p = beg; p < end; p++) {
                    uint32_t local = assignments[p];
                    if (local == std::numeric_limits<uint32_t>::max() || local >= kb0) local = 0;
                    cnt[(size_t)local]++;
                }
                for (uint32_t c = 0; c < kb0; c++) sub_sizes[(size_t)(sub0 + c)] = (uint64_t)cnt[(size_t)c];

                // centroids: copy from existing centroids (global clusters are contiguous per bucket)
                uint64_t coff = cluster_offsets[b];
                std::memcpy(&sub_centroids[(size_t)sub0 * (size_t)D],
                            cent_out + coff * (uint64_t)D,
                            (size_t)kb0 * (size_t)D * sizeof(float));
            }
        }

        // prefix-sum -> sub_offsets
        std::vector<uint64_t> sub_offsets((size_t)n_sub + 1, 0);
        for (uint64_t s = 0; s < n_sub; s++) {
            sub_offsets[(size_t)s + 1] = sub_offsets[(size_t)s] + sub_sizes[(size_t)s];
        }
        if (sub_offsets.back() != N) {
            die("sub_offsets total mismatch: sub_offsets[n_sub]=" + std::to_string(sub_offsets.back()) +
                " but N=" + std::to_string(N));
        }

        // write sub_offsets + bucket_sub_offsets
        {
            std::ofstream f(p_sub_offsets, std::ios::binary | std::ios::trunc);
            f.write((const char*)sub_offsets.data(), (std::streamsize)sub_offsets.size() * (std::streamsize)sizeof(uint64_t));
            f.flush();
        }
        {
            std::ofstream f(p_bsub_off, std::ios::binary | std::ios::trunc);
            f.write((const char*)bucket_sub_offsets.data(), (std::streamsize)bucket_sub_offsets.size() * (std::streamsize)sizeof(uint64_t));
            f.flush();
        }

        // write meta + centroids
        {
            std::ofstream f(p_sub_bid, std::ios::binary | std::ios::trunc);
            f.write((const char*)sub_bucket_id.data(), (std::streamsize)sub_bucket_id.size() * (std::streamsize)sizeof(uint32_t));
            f.flush();
        }
        if (a.sub_cluster_id_u16) {
            std::ofstream f(p_sub_cid16, std::ios::binary | std::ios::trunc);
            f.write((const char*)sub_cluster_id_u16.data(), (std::streamsize)sub_cluster_id_u16.size() * (std::streamsize)sizeof(uint16_t));
            f.flush();
        } else {
            std::ofstream f(p_sub_cid32, std::ios::binary | std::ios::trunc);
            f.write((const char*)sub_cluster_id_u32.data(), (std::streamsize)sub_cluster_id_u32.size() * (std::streamsize)sizeof(uint32_t));
            f.flush();
        }
        {
            std::ofstream f(p_sub_cent, std::ios::binary | std::ios::trunc);
            f.write((const char*)sub_centroids.data(), (std::streamsize)sub_centroids.size() * (std::streamsize)sizeof(float));
            f.flush();
        }

        // Pass2: fill sub_ids
        MMap mm_sub_ids;
        mm_sub_ids.open_rw_create(p_sub_ids, (size_t)(N * sizeof(int64_t)));
        int64_t* sub_ids = (int64_t*)mm_sub_ids.addr;

        double t_sub1 = now_sec();

#pragma omp parallel for schedule(dynamic, 256)
        for (int64_t bi = 0; bi < (int64_t)nlist; bi++) {
            uint64_t b = (uint64_t)bi;
            uint64_t beg = offsets[b], end = offsets[b+1];
            uint64_t sz = end - beg;

            uint64_t sub0 = bucket_sub_offsets[b];
            uint32_t kb0 = k_per_bucket[b];

            if (kb0 == 0) {
                // copy ids in bucket order
                uint64_t w = sub_offsets[(size_t)sub0];
                for (uint64_t p = beg; p < end; p++) {
                    sub_ids[w++] = (int64_t)bids[p];
                }
            } else {
                std::vector<uint64_t> wp((size_t)kb0, 0);
                for (uint32_t c = 0; c < kb0; c++) wp[(size_t)c] = sub_offsets[(size_t)sub0 + (size_t)c];

                for (uint64_t p = beg; p < end; p++) {
                    uint32_t local = assignments[p];
                    if (local == std::numeric_limits<uint32_t>::max() || local >= kb0) local = 0;
                    uint64_t dst = wp[(size_t)local]++;
                    sub_ids[dst] = (int64_t)bids[p];
                }

                // (optional) sanity check in debug builds
#ifndef NDEBUG
                for (uint32_t c = 0; c < kb0; c++) {
                    uint64_t expect_end = sub_offsets[(size_t)sub0 + (size_t)c + 1];
                    if (wp[(size_t)c] != expect_end) {
                        die("sub_ids write ptr mismatch at bucket=" + std::to_string(b) + " c=" + std::to_string(c));
                    }
                }
#endif
            }
        }

        msync(mm_sub_ids.addr, mm_sub_ids.len, MS_SYNC);

        double sec_p1 = elapsed_sec(t_sub0);
        double sec_p2 = elapsed_sec(t_sub1);
        std::cerr << "sub-CSR pass1 (counts/meta/centroids) : " << sec_p1 << " sec\n";
        std::cerr << "sub-CSR pass2 (fill ids)              : " << sec_p2 << " sec\n";
        std::cerr << "sub-CSR outputs:\n"
                  << "  " << p_sub_offsets << "\n"
                  << "  " << p_sub_ids << "\n"
                  << "  " << p_sub_bid << "\n"
                  << "  " << (a.sub_cluster_id_u16 ? p_sub_cid16 : p_sub_cid32) << "\n"
                  << "  " << p_sub_cent << "\n"
                  << "  " << p_bsub_off << "\n";
    }

    // -------------------- GT ANALYSIS (optional) --------------------
    if (want_gt) {
        std::cerr << "\n==== GT ANALYSIS ====\n";

        // Load queries
        MMap mm_q;
        const float* xq = nullptr;
        uint64_t Q = 0;
        uint32_t Dq = 0;

        if (!q_path_npy.empty()) {
            NpyArrayView qv = load_npy_view(mm_q, q_path_npy);
            if (qv.descr != "<f4") die("[GT] q_npy dtype mismatch: got " + qv.descr + " want <f4");
            if (qv.shape.size() != 2) die("[GT] q_npy must be 2D [Q,D]");
            Q = qv.shape[0];
            Dq = (uint32_t)qv.shape[1];
            xq = (const float*)qv.data;
        } else {
            Q = a.Q_arg;
            Dq = a.Dq_arg;
            uint64_t bytes = file_size(q_path_bin);
            uint64_t expect = Q * (uint64_t)Dq * sizeof(float);
            if (bytes != expect) die("[GT] q_f32_bin size mismatch: got=" + std::to_string(bytes) + " expect=" + std::to_string(expect));
            mm_q.open_ro(q_path_bin);
            xq = (const float*)mm_q.addr;
        }

        if (Dq != D) die("[GT] query dim Dq must equal xb/centroids D: Dq=" + std::to_string(Dq) + " D=" + std::to_string(D));

        // Parse Q,K from gt filenames
        uint64_t QI=0, QD=0;
        uint32_t KI=0, KD=0;
        if (!parse_QK_from_gt_filename(gt_I_path, &QI, &KI)) die("[GT] cannot parse Q,K from gt_I filename: " + gt_I_path);
        if (QI != Q) die("[GT] Q mismatch: query Q=" + std::to_string(Q) + " gt_I Q=" + std::to_string(QI));
        uint32_t K = KI;

        bool have_gtD = false;
        if (!gt_D_path.empty()) {
            if (!parse_QK_from_gt_filename(gt_D_path, &QD, &KD)) die("[GT] cannot parse Q,K from gt_D filename: " + gt_D_path);
            if (QD != Q) die("[GT] Q mismatch: query Q=" + std::to_string(Q) + " gt_D Q=" + std::to_string(QD));
            if (KD != K) die("[GT] K mismatch: gt_I K=" + std::to_string(K) + " gt_D K=" + std::to_string(KD));
            have_gtD = true;
        }

        // mmap gt bins
        MMap mm_gtI, mm_gtD;
        mm_gtI.open_ro(gt_I_path);
        uint64_t bytesI = mm_gtI.len;
        if (bytesI != Q * (uint64_t)K * 8ULL) {
            die("[GT] gt_I file bytes mismatch: got=" + std::to_string(bytesI) +
                " expect=" + std::to_string(Q*(uint64_t)K*8ULL));
        }
        const int64_t* gtI = (const int64_t*)mm_gtI.addr;

        const float* gtD = nullptr;
        if (have_gtD) {
            mm_gtD.open_ro(gt_D_path);
            uint64_t bytesD = mm_gtD.len;
            if (bytesD != Q * (uint64_t)K * 4ULL) {
                die("[GT] gt_D file bytes mismatch: got=" + std::to_string(bytesD) +
                    " expect=" + std::to_string(Q*(uint64_t)K*4ULL));
            }
            gtD = (const float*)mm_gtD.addr;
        }

        std::cerr << "[GT] loaded Q=" << Q << " K=" << K << " D=" << D << "\n";
        std::cerr << "[GT] building id2pos (uint32[N]) ... (~" << (double)N*4.0/(1024.0*1024.0*1024.0) << " GiB)\n";

        // id2pos: original id -> bucketed position
        std::vector<uint32_t> id2pos;
        try {
            id2pos.assign((size_t)N, std::numeric_limits<uint32_t>::max());
        } catch (...) {
            die("[GT] failed to allocate id2pos (need ~4GB for N=1B). Try smaller N or run on big-mem node.");
        }

#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)N; i++) {
            uint32_t id = bids[(uint64_t)i];
            id2pos[(size_t)id] = (uint32_t)i;
        }

        std::vector<uint32_t> pos2bucket;
        if (a.gt_build_pos2bucket) {
            std::cerr << "[GT] building pos2bucket (uint32[N]) ... (~" << (double)N*4.0/(1024.0*1024.0*1024.0) << " GiB)\n";
            try {
                pos2bucket.assign((size_t)N, 0);
            } catch (...) {
                die("[GT] failed to allocate pos2bucket. Set --gt_build_pos2bucket 0 to save ~4GB.");
            }
#pragma omp parallel for schedule(dynamic, 1024)
            for (int64_t bi = 0; bi < (int64_t)nlist; bi++) {
                uint64_t b = (uint64_t)bi;
                uint64_t beg = offsets[b], end = offsets[b+1];
                for (uint64_t p = beg; p < end; p++) {
                    pos2bucket[(size_t)p] = (uint32_t)b;
                }
            }
        }

        auto find_bucket_by_offsets = [&](uint64_t pos)->uint32_t {
            // Fallback if pos2bucket disabled: binary search offsets to find b such that offsets[b] <= pos < offsets[b+1]
            auto it = std::upper_bound(offsets, offsets + nlist + 1, pos);
            if (it == offsets) return 0;
            uint64_t b = (uint64_t)(it - offsets - 1);
            if (b >= nlist) b = nlist - 1;
            return (uint32_t)b;
        };

        // Optional CSV dump
        std::ofstream dump;
        if (a.gt_dump_cap_per_query > 0) {
            dump.open(a.out_dir + "/gt_cluster_distance_pairs.csv");
            dump << "qid,k,gt_id,bucket,local_cluster,global_cluster,"
                    "d_q_gt_recompute_l2,d_q_gt_recompute_l2sqr,"
                    "d_q_gt_given_raw,d_q_gt_given_l2,d_q_gt_given_l2sqr,"
                    "d_q_cent_l2,d_q_cent_l2sqr,"
                    "diff_l2,ratio_l2,diff_l2sqr,ratio_l2sqr\n";
            dump << std::fixed << std::setprecision(6);
        }

        // Reservoirs for L2 domain (primary)
        Reservoir r_dgt((size_t)a.gt_summary_sample, a.seed ^ 0x12345678ULL);
        Reservoir r_dcent((size_t)a.gt_summary_sample, a.seed ^ 0x9abcdef0ULL);
        Reservoir r_diff((size_t)a.gt_summary_sample, a.seed ^ 0xdeadbeefULL);
        Reservoir r_ratio((size_t)a.gt_summary_sample, a.seed ^ 0x31415926ULL);

        // Reservoirs for L2^2 domain
        Reservoir r_dgt2((size_t)a.gt_summary_sample, a.seed ^ 0x11111111ULL);
        Reservoir r_dcent2((size_t)a.gt_summary_sample, a.seed ^ 0x22222222ULL);
        Reservoir r_diff2((size_t)a.gt_summary_sample, a.seed ^ 0x33333333ULL);
        Reservoir r_ratio2((size_t)a.gt_summary_sample, a.seed ^ 0x44444444ULL);

        // Linear relationship accumulators (given vs recompute)
        LinAcc acc_l2, acc_l2sqr;
        std::mutex acc_mu;

        double t_gt = now_sec();
        std::atomic<uint64_t> pairs_done{0};

        // Use parallel region with per-thread accumulators to reduce lock contention.
#pragma omp parallel
        {
            LinAcc local_l2, local_l2sqr;

#pragma omp for schedule(dynamic, 32)
            for (int64_t qi = 0; qi < (int64_t)Q; qi++) {
                const float* qv = xq + (uint64_t)qi * (uint64_t)D;

                int dump_cap = a.gt_dump_cap_per_query;

                for (uint32_t kk = 0; kk < K; kk++) {
                    int64_t id64 = gtI[(uint64_t)qi * (uint64_t)K + kk];
                    if (id64 < 0) continue;
                    uint64_t id = (uint64_t)id64;
                    if (id >= N) continue;

                    uint32_t pos = id2pos[(size_t)id];
                    if (pos == std::numeric_limits<uint32_t>::max()) continue;

                    uint32_t b = a.gt_build_pos2bucket ? pos2bucket[(size_t)pos] : find_bucket_by_offsets((uint64_t)pos);

                    uint32_t local = assignments[(uint64_t)pos];
                    if (local == std::numeric_limits<uint32_t>::max()) continue;

                    uint64_t coff = cluster_offsets[b];
                    uint64_t global_cluster = coff + (uint64_t)local;
                    if (global_cluster >= totalK) continue;

                    const float* cent = cent_out + global_cluster * (uint64_t)D;

                    // centroid distances (both domains)
                    float d_cent_l2sqr = l2sqr(qv, cent, D);
                    float d_cent_l2    = std::sqrt(std::max(0.0f, d_cent_l2sqr));

                    // given gt_D (raw + normalized into both domains)
                    float d_gt_given_raw   = std::numeric_limits<float>::quiet_NaN();
                    float d_gt_given_l2    = std::numeric_limits<float>::quiet_NaN();
                    float d_gt_given_l2sqr = std::numeric_limits<float>::quiet_NaN();

                    if (have_gtD) {
                        d_gt_given_raw = gtD[(uint64_t)qi * (uint64_t)K + kk];
                        if (std::isfinite(d_gt_given_raw)) {
                            if (a.gtD_is_l2sqr) {
                                d_gt_given_l2sqr = d_gt_given_raw;
                                d_gt_given_l2 = std::sqrt(std::max(0.0f, d_gt_given_l2sqr));
                            } else {
                                d_gt_given_l2 = d_gt_given_raw;
                                d_gt_given_l2sqr = d_gt_given_l2 * d_gt_given_l2;
                            }
                        }
                    }

                    // recompute q-gt (both domains)
                    float d_gt_re_l2    = std::numeric_limits<float>::quiet_NaN();
                    float d_gt_re_l2sqr = std::numeric_limits<float>::quiet_NaN();
                    if (a.gt_recompute_q_gt) {
                        const float* xv = xb + id * (uint64_t)D;
                        d_gt_re_l2sqr = l2sqr(qv, xv, D);
                        d_gt_re_l2 = std::sqrt(std::max(0.0f, d_gt_re_l2sqr));
                    }

                    // choose a "primary" gt distance (prefer recompute, else given)
                    bool have_primary = false;
                    float d_gt_l2 = std::numeric_limits<float>::quiet_NaN();
                    float d_gt_l2sqr = std::numeric_limits<float>::quiet_NaN();

                    if (a.gt_recompute_q_gt && std::isfinite(d_gt_re_l2)) {
                        d_gt_l2 = d_gt_re_l2;
                        d_gt_l2sqr = d_gt_re_l2sqr;
                        have_primary = true;
                    } else if (have_gtD && std::isfinite(d_gt_given_l2)) {
                        d_gt_l2 = d_gt_given_l2;
                        d_gt_l2sqr = d_gt_given_l2sqr;
                        have_primary = true;
                    }
                    if (!have_primary) continue;

                    float diff_l2 = d_gt_l2 - d_cent_l2;
                    float ratio_l2 = d_gt_l2 / (d_cent_l2 + 1e-12f);

                    float diff_l2sqr = d_gt_l2sqr - d_cent_l2sqr;
                    float ratio_l2sqr = d_gt_l2sqr / (d_cent_l2sqr + 1e-20f);

                    r_dgt.push(d_gt_l2);
                    r_dcent.push(d_cent_l2);
                    r_diff.push(diff_l2);
                    r_ratio.push(ratio_l2);

                    r_dgt2.push(d_gt_l2sqr);
                    r_dcent2.push(d_cent_l2sqr);
                    r_diff2.push(diff_l2sqr);
                    r_ratio2.push(ratio_l2sqr);

                    // If both given and recompute exist, accumulate linear fit stats:
                    if (have_gtD && a.gt_recompute_q_gt &&
                        std::isfinite(d_gt_given_l2) && std::isfinite(d_gt_re_l2)) {

                        local_l2.add((double)d_gt_given_l2, (double)d_gt_re_l2);
                        local_l2sqr.add((double)d_gt_given_l2sqr, (double)d_gt_re_l2sqr);
                    }

                    // dump (best-effort)
                    if (dump_cap > 0 && kk < (uint32_t)dump_cap) {
#pragma omp critical
                        {
                            dump << qi << "," << kk << "," << id << "," << b << "," << local << "," << global_cluster << ","
                                 << (std::isfinite(d_gt_re_l2) ? d_gt_re_l2 : std::numeric_limits<float>::quiet_NaN()) << ","
                                 << (std::isfinite(d_gt_re_l2sqr) ? d_gt_re_l2sqr : std::numeric_limits<float>::quiet_NaN()) << ","
                                 << (have_gtD ? d_gt_given_raw : std::numeric_limits<float>::quiet_NaN()) << ","
                                 << (std::isfinite(d_gt_given_l2) ? d_gt_given_l2 : std::numeric_limits<float>::quiet_NaN()) << ","
                                 << (std::isfinite(d_gt_given_l2sqr) ? d_gt_given_l2sqr : std::numeric_limits<float>::quiet_NaN()) << ","
                                 << d_cent_l2 << "," << d_cent_l2sqr << ","
                                 << diff_l2 << "," << ratio_l2 << ","
                                 << diff_l2sqr << "," << ratio_l2sqr << "\n";
                        }
                    }

                    pairs_done.fetch_add(1, std::memory_order_relaxed);
                }
            } // end for qi

            // merge thread accumulators
            {
                std::lock_guard<std::mutex> lk(acc_mu);
                acc_l2.merge(local_l2);
                acc_l2sqr.merge(local_l2sqr);
            }
        } // end omp parallel

        double sec_gt = elapsed_sec(t_gt);
        std::cerr << "[GT] done in " << sec_gt << " sec, pairs_done=" << pairs_done.load() << "\n";

        // Write base summary (L2 domain)
        std::string out_sum = a.out_dir + "/gt_cluster_distance_summary.txt";
        write_gt_summary_txt_base(
            out_sum,
            Q, K,
            r_dgt.buf, r_dcent.buf, r_diff.buf, r_ratio.buf,
            sec_gt, pairs_done.load(),
            have_gtD, a.gt_recompute_q_gt != 0,
            a.gtD_is_l2sqr
        );

        // Append linear-fit stats + L2^2 domain summaries
        double slope=0, intercept=0, corr=0, rmse=0;
        double slope2=0, intercept2=0, corr2=0, rmse2=0;
        lin_finalize(acc_l2, &slope, &intercept, &corr, &rmse);
        lin_finalize(acc_l2sqr, &slope2, &intercept2, &corr2, &rmse2);

        {
            std::ofstream out(out_sum, std::ios::app);

            out << "\nLINEAR_FIT (x=given, y=recompute)\n";
            out << "L2:   n=" << acc_l2.n
                << " slope=" << std::fixed << std::setprecision(8) << slope
                << " intercept=" << intercept
                << " corr=" << corr
                << " rmse=" << rmse
                << "\n";

            out << "L2^2: n=" << acc_l2sqr.n
                << " slope=" << slope2
                << " intercept=" << intercept2
                << " corr=" << corr2
                << " rmse=" << rmse2
                << "\n";

            auto dump_quant = [&](const std::string& name, std::vector<float> v) {
                std::vector<float> tmp = std::move(v);
                float p50 = percentile_from_sample(tmp, 0.50f);
                float p90 = percentile_from_sample(tmp, 0.90f);
                float p99 = percentile_from_sample(tmp, 0.99f);
                out << name << "_p50=" << p50 << " p90=" << p90 << " p99=" << p99 << " n=" << tmp.size() << "\n";
            };

            if (a.gt_report_both_domains) {
                out << "\nDOMAIN_L2SQR_STATS\n";
                dump_quant("d_q_gt_l2sqr(primary)", r_dgt2.buf);
                dump_quant("d_q_cent_l2sqr", r_dcent2.buf);
                dump_quant("diff_l2sqr(d_gt-d_cent)", r_diff2.buf);
                dump_quant("ratio_l2sqr(d_gt/(d_cent+eps))", r_ratio2.buf);
            }
        }

        std::cerr << "[GT] summary -> " << out_sum << "\n";
        if (a.gt_dump_cap_per_query > 0) std::cerr << "[GT] csv -> " << (a.out_dir + "/gt_cluster_distance_pairs.csv") << "\n";
    }

    std::cerr << "Outputs in: " << a.out_dir << "\n";
    return 0;
}
