// bench_enum_centroid_stage_only_v6_8_calibcsv_only_clean.cpp
// Cleaned calib_csv-only single-file version for pipeline reintegration.
// Key changes vs your current test file:
//   1) FIXED heap IDs: heap now stores REAL sub-centroid IDs (no fake placeholder IDs)
//   2) Removed heavy analysis/diagnostic stats & verbose outputs
//   3) Kept only the calib_csv copy-path + singleton split + early-stop logic
//
// Dependencies:
//   - csr_index_runtime_v3_with_subcsr.h
//   - query_bins_bins_only_v3a_standalone.hpp
//   - resrank_kernels.h (included for compatibility)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <sys/stat.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "csr_index_runtime_v3_with_subcsr.h"
#include "query_bins_bins_only_v3a_standalone.hpp"
#include "resrank_kernels.h"

using WallClock = std::chrono::steady_clock;

// ---------------- basic utils ----------------
static inline double ms_since(const WallClock::time_point& t0){
    return std::chrono::duration<double, std::milli>(WallClock::now() - t0).count();
}

[[noreturn]] static inline void die(const std::string& s){
    std::cerr << "[FATAL] " << s << "\n";
    std::exit(1);
}

static inline bool file_exists_fs(const std::string& p) {
    struct stat st; return (::stat(p.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}
static inline bool dir_exists_fs(const std::string& p) {
    struct stat st; return (::stat(p.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}
static inline void ensure_dir_fs(const std::string& p) {
    if (p.empty() || dir_exists_fs(p)) return;
    ::mkdir(p.c_str(), 0755);
}
static inline void copy_file_bytes(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary); if (!in) die("copy_file_bytes: cannot open src: " + src);
    std::ofstream out(dst, std::ios::binary | std::ios::trunc); if (!out) die("copy_file_bytes: cannot open dst: " + dst);
    out << in.rdbuf(); if (!out) die("copy_file_bytes: write failed: " + dst);
}
static inline void ensure_symlink_or_copy(const std::string& src, const std::string& dst) {
    if (src.empty() || dst.empty() || src == dst) return;
    if (!file_exists_fs(src)) die("ensure_symlink_or_copy: missing src: " + src);
    auto pos = dst.find_last_of('/');
    if (pos != std::string::npos) ensure_dir_fs(dst.substr(0, pos));
    if (file_exists_fs(dst)) return;
    if (::symlink(src.c_str(), dst.c_str()) == 0) return;
    copy_file_bytes(src, dst);
}
static inline std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    return (a.back() == '/') ? (a + b) : (a + "/" + b);
}
static inline std::string pick_path_or_default(const std::string& explicit_path, const std::string& dir, const std::string& filename) {
    if (!explicit_path.empty()) return explicit_path;
    if (dir.empty()) return filename;
    return path_join(dir, filename);
}

template<typename T>
static inline std::vector<T> read_bin_or_die(const std::string& path, size_t expect_n=0){
    std::ifstream f(path, std::ios::binary);
    if(!f) die("cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t nbytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if(nbytes % sizeof(T)) die("bad size (not multiple of dtype): " + path);
    size_t n = nbytes / sizeof(T);
    if(expect_n && n != expect_n) {
        std::ostringstream oss; oss << "size mismatch " << path << " got " << n << " expected " << expect_n;
        die(oss.str());
    }
    std::vector<T> v(n);
    if(n) f.read((char*)v.data(), (std::streamsize)nbytes);
    if(!f) die("read failed: " + path);
    return v;
}

// ---------------- simple JSON extractors ----------------
static inline bool json_find_key_pos(const std::string& s, const std::string& key, size_t& kpos){
    std::string pat = "\"" + key + "\"";
    kpos = s.find(pat);
    return kpos != std::string::npos;
}
static inline bool json_find_value_start(const std::string& s, size_t kpos, size_t& vpos){
    size_t c = s.find(':', kpos);
    if(c == std::string::npos) return false;
    vpos = c + 1;
    while(vpos < s.size() && std::isspace((unsigned char)s[vpos])) ++vpos;
    return vpos < s.size();
}
static inline bool json_extract_string(const std::string& s, const std::string& key, std::string& out){
    size_t k,v; if(!json_find_key_pos(s,key,k)) return false;
    if(!json_find_value_start(s,k,v)) return false;
    if(s[v] != '"') return false;
    ++v;
    std::string r;
    while(v < s.size()){
        char ch = s[v++];
        if(ch == '\\' && v < s.size()){
            char e = s[v++];
            switch(e){
                case '"': r.push_back('"'); break;
                case '\\': r.push_back('\\'); break;
                case '/': r.push_back('/'); break;
                case 'b': r.push_back('\b'); break;
                case 'f': r.push_back('\f'); break;
                case 'n': r.push_back('\n'); break;
                case 'r': r.push_back('\r'); break;
                case 't': r.push_back('\t'); break;
                default: r.push_back(e); break;
            }
            continue;
        }
        if(ch == '"'){ out = r; return true; }
        r.push_back(ch);
    }
    return false;
}
template<typename T>
static inline bool json_extract_num(const std::string& s, const std::string& key, T& out){
    size_t k,v; if(!json_find_key_pos(s,key,k)) return false;
    if(!json_find_value_start(s,k,v)) return false;
    size_t e=v;
    while(e < s.size()){
        char ch=s[e];
        if(std::isdigit((unsigned char)ch)||ch=='-'||ch=='+'||ch=='.'||ch=='e'||ch=='E') { ++e; continue; }
        break;
    }
    if(e<=v) return false;
    std::stringstream ss(s.substr(v,e-v));
    T t{}; ss >> t;
    if(ss.fail()) return false;
    out=t; return true;
}
static inline bool json_extract_bool(const std::string& s, const std::string& key, bool& out){
    size_t k,v;
    if(!json_find_key_pos(s,key,k)) return false;
    if(!json_find_value_start(s,k,v)) return false;
    if(s.compare(v,4,"true")==0){ out=true; return true; }
    if(s.compare(v,5,"false")==0){ out=false; return true; }
    return false;
}
static inline std::string slurp(const std::string& p){
    std::ifstream f(p);
    if(!f) die("cannot open json: " + p);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

#ifdef _OPENMP
static inline void omp_apply_schedule(const std::string& sched, int chunk){
    omp_sched_t kind = omp_sched_static;
    if(sched=="dynamic") kind = omp_sched_dynamic;
    else if(sched=="guided") kind = omp_sched_guided;
    else if(sched=="auto") kind = omp_sched_auto;
    omp_set_dynamic(0);
    omp_set_schedule(kind, std::max(0,chunk));
}
#endif

static inline void set_env_if_nonempty(const std::string& k, const std::string& v){
    if(!v.empty()) ::setenv(k.c_str(), v.c_str(), 1);
}

static inline double pct_from_vec(std::vector<double> v, double p){
    if(v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double x=(p/100.0)*(double)(v.size()-1);
    size_t i=(size_t)std::floor(x);
    size_t j=std::min(v.size()-1,i+1);
    double a=x-(double)i;
    return v[i]*(1.0-a)+v[j]*a;
}

// ---------------- heap ----------------
enum class HeapPushOutcome : uint8_t {
    Rejected = 0,
    Appended = 1,
    Replaced = 2
};

struct TopKHeap {
    int K=0;
    std::vector<float> d;        // max-heap by distance (worst at root)
    std::vector<uint32_t> cid;   // REAL sub-centroid ids
    float best_seen_cached = std::numeric_limits<float>::infinity();

    void reset(int k){
        K=k;
        d.clear(); cid.clear();
        best_seen_cached = std::numeric_limits<float>::infinity();
        if(k>0){ d.reserve((size_t)k); cid.reserve((size_t)k); }
    }
    inline bool full() const { return K>0 && (int)d.size()>=K; }
    inline int size() const { return (int)d.size(); }
    inline float worst() const { return d.empty() ? std::numeric_limits<float>::infinity() : d[0]; }
    inline float best_linear() const { return best_seen_cached; }

    inline HeapPushOutcome push_with_outcome(float dist, uint32_t id){
        if(K<=0) return HeapPushOutcome::Rejected;
        if((int)d.size() < K){
            d.push_back(dist); cid.push_back(id);
            if(dist < best_seen_cached) best_seen_cached = dist;
            if((int)d.size() == K) heapify();
            return HeapPushOutcome::Appended;
        }
        if(dist >= d[0]) return HeapPushOutcome::Rejected;
        d[0]=dist; cid[0]=id; sift_down(0);
        if(dist < best_seen_cached) best_seen_cached = dist;
        return HeapPushOutcome::Replaced;
    }

    inline void heapify(){ for(int i=((int)d.size()/2)-1;i>=0;--i) sift_down(i); }
    inline void sift_down(int i){
        const int n=(int)d.size();
        while(true){
            int l=2*i+1, r=l+1, m=i;
            if(l<n && d[l]>d[m]) m=l;
            if(r<n && d[r]>d[m]) m=r;
            if(m==i) break;
            std::swap(d[i],d[m]);
            std::swap(cid[i],cid[m]);
            i=m;
        }
    }
};

// ---------------- CSV calibration ----------------
struct CalibMarginRow {
    float dbest2_knot = 0.0f;
    int   n_samples = 0;
    float window_halfwidth = 0.0f;
    float q95_raw = 0.0f;
    float q99_raw = 0.0f;
    float q95_use = 0.0f;
    float q99_use = 0.0f;
};

struct CalibMarginTable {
    std::vector<CalibMarginRow> rows;
    bool loaded = false;
    std::string source_csv;
    std::string quantile = "q99"; // "q95" or "q99"

    float margin_from_best2(float dbest2) const {
        if(rows.empty()) return 0.0f;
        if(!std::isfinite(dbest2)) {
            const auto& r0 = rows.front();
            return (quantile == "q95") ? r0.q95_use : r0.q99_use;
        }
        if(dbest2 <= rows.front().dbest2_knot) {
            const auto& r = rows.front();
            return (quantile == "q95") ? r.q95_use : r.q99_use;
        }
        if(dbest2 >= rows.back().dbest2_knot) {
            const auto& r = rows.back();
            return (quantile == "q95") ? r.q95_use : r.q99_use;
        }
        size_t lo = 0, hi = rows.size() - 1;
        while(lo + 1 < hi) {
            size_t mid = (lo + hi) >> 1;
            if(rows[mid].dbest2_knot <= dbest2) lo = mid;
            else hi = mid;
        }
        const auto& a = rows[lo];
        const auto& b = rows[hi];
        const float xa = a.dbest2_knot;
        const float xb = b.dbest2_knot;
        const float ya = (quantile == "q95") ? a.q95_use : a.q99_use;
        const float yb = (quantile == "q95") ? b.q95_use : b.q99_use;
        if(xb <= xa) return ya;
        const float t = (dbest2 - xa) / (xb - xa);
        return ya + t * (yb - ya);
    }
};

static inline std::string trim_copy(const std::string& s){
    size_t i = 0, j = s.size();
    while(i < j && std::isspace((unsigned char)s[i])) ++i;
    while(j > i && std::isspace((unsigned char)s[j-1])) --j;
    return s.substr(i, j-i);
}
static inline std::vector<std::string> split_csv_line_simple(const std::string& line){
    std::vector<std::string> out;
    std::string cur;
    for(char c : line){
        if(c == ','){ out.push_back(trim_copy(cur)); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(trim_copy(cur));
    return out;
}
static inline float parse_f_or_die(const std::string& s, const std::string& what){
    char* endp = nullptr;
    float v = std::strtof(s.c_str(), &endp);
    if(endp == s.c_str()) die("failed to parse float for " + what + ": " + s);
    return v;
}
static inline int parse_i_or_die(const std::string& s, const std::string& what){
    char* endp = nullptr;
    long v = std::strtol(s.c_str(), &endp, 10);
    if(endp == s.c_str()) die("failed to parse int for " + what + ": " + s);
    if(v < (long)std::numeric_limits<int>::min() || v > (long)std::numeric_limits<int>::max())
        die("int out of range for " + what + ": " + s);
    return (int)v;
}
static inline CalibMarginTable load_calib_margin_csv_or_die(
    const std::string& csv_path,
    const std::string& quantile)
{
    if(csv_path.empty()) die("calib margin csv path is empty");
    std::ifstream f(csv_path);
    if(!f) die("cannot open calib margin csv: " + csv_path);

    std::string header;
    if(!std::getline(f, header)) die("empty calib margin csv: " + csv_path);
    auto cols = split_csv_line_simple(header);

    std::unordered_map<std::string, int> col;
    for(int i=0; i<(int)cols.size(); ++i) col[cols[i]] = i;

    auto need_col = [&](const std::string& name){
        if(col.find(name) == col.end()) die("calib csv missing column: " + name);
        return col[name];
    };

    const int c_dbest2 = need_col("dbest2_knot");
    const int c_ns     = need_col("n_samples");
    const int c_wh     = need_col("window_halfwidth");
    const int c_q95r   = need_col("q95_raw");
    const int c_q99r   = need_col("q99_raw");
    const int c_q95u   = need_col("q95_use");
    const int c_q99u   = need_col("q99_use");

    CalibMarginTable tab;
    tab.source_csv = csv_path;
    tab.quantile = (quantile == "q95" ? "q95" : "q99");

    std::string line;
    int lineno = 1;
    while(std::getline(f, line)){
        ++lineno;
        line = trim_copy(line);
        if(line.empty()) continue;
        if(line[0] == '#') continue;

        auto v = split_csv_line_simple(line);
        const int need_n = std::max({c_dbest2,c_ns,c_wh,c_q95r,c_q99r,c_q95u,c_q99u}) + 1;
        if((int)v.size() < need_n){
            std::ostringstream oss; oss << "calib csv bad row (too few cols) line=" << lineno;
            die(oss.str());
        }

        CalibMarginRow r;
        r.dbest2_knot      = parse_f_or_die(v[c_dbest2], "dbest2_knot");
        r.n_samples        = parse_i_or_die(v[c_ns], "n_samples");
        r.window_halfwidth = parse_f_or_die(v[c_wh], "window_halfwidth");
        r.q95_raw          = parse_f_or_die(v[c_q95r], "q95_raw");
        r.q99_raw          = parse_f_or_die(v[c_q99r], "q99_raw");
        r.q95_use          = parse_f_or_die(v[c_q95u], "q95_use");
        r.q99_use          = parse_f_or_die(v[c_q99u], "q99_use");
        tab.rows.push_back(r);
    }

    if(tab.rows.empty()) die("calib csv has no data rows: " + csv_path);

    std::sort(tab.rows.begin(), tab.rows.end(),
              [](const CalibMarginRow& a, const CalibMarginRow& b){
                  return a.dbest2_knot < b.dbest2_knot;
              });

    tab.loaded = true;
    return tab;
}

// ---------------- config ----------------
struct Config {
    std::string thr_f32_path;
    std::string logits_path;
    std::string metric = "l2_u8_128";
    int D_logits = 0, Q = 0, D_in = 0;
    int repeats = 1, omp_threads = 0;

    int Lsel = 16;
    float Rmax = 7.0f;
    int NBINS = 256;
    int cap_per_bin = 256;

    std::string sched_enum = "dynamic";
    int chunk_enum = 32;
    std::string omp_bind, omp_places;

    std::string subcsr_meta_dir, subcsr_cluster_dir;
    std::string sub_ids_path, sub_offsets_path, sub_bucket_id_path, sub_cluster_id_path, bucket_sub_offsets_path, sub_centroids_path;
    bool sub_cluster_u16 = true;
    uint64_t csr_N = 0;
    uint32_t csr_nlist = 0;
    int sub_centroids_d = 0, sub_codes_d = 0;

    float max_score = std::numeric_limits<float>::infinity();
    bool keep_base = true;

    int warmup_enum_iters = 1;
    int warmup_q = 256;
    bool warmup_random = true;
    bool warmup_use_real_logits = false;
    int random_seed = 12345;

    // centroid copy path
    int centroid_heap_k = 1024;
    int centroid_kernel_batch_clusters = 4096;
    bool merge_adjacent_spans = true;
    bool sort_bucket_ids_for_centroid = false;
    std::string centroid_batch_mode = "copy";

    // singleton split
    bool singleton_direct_refine = true;

    // calib csv gate
    bool centroid_early_stop = true;
    int centroid_early_stop_after_bins = 4;
    std::string centroid_gate_mode = "calib_csv";
    std::string calib_margin_csv;
    std::string calib_margin_quantile = "q99";

    // legacy fallback quality criteria
    float calib_batch_qual_ratio_eps = 0.0f;
    int   calib_batch_qual_abs_eps = 0;
    int   calib_min_batches_before_stop = 3;

    // gate option
    int   heap_gate_by_calib_csv = 0;

    // heap-driven low-quality criteria
    int   calib_lowqual_use_heap_updates = 1;
    float calib_heap_front_alpha = 0.50f;
    float calib_heap_front_update_ratio_eps = 0.0f;
    int   calib_heap_front_update_abs_eps = 0;
    int   calib_heap_updates_abs_eps = 0;
};

static inline void apply_json(Config& cfg, const std::string& js){
    json_extract_string(js,"thr_f32_path", cfg.thr_f32_path);
    json_extract_string(js,"logits_path", cfg.logits_path);
    json_extract_string(js,"metric", cfg.metric);

    json_extract_num(js,"D", cfg.D_logits);
    json_extract_num(js,"Q", cfg.Q);
    json_extract_num(js,"D_in", cfg.D_in);
    json_extract_num(js,"repeats", cfg.repeats);
    json_extract_num(js,"omp_threads", cfg.omp_threads);

    json_extract_num(js,"Lsel", cfg.Lsel);
    json_extract_num(js,"Rmax", cfg.Rmax);
    json_extract_num(js,"NBINS", cfg.NBINS);
    json_extract_num(js,"cap_per_bin", cfg.cap_per_bin);

    json_extract_string(js,"sched_enum", cfg.sched_enum);
    json_extract_num(js,"chunk_enum", cfg.chunk_enum);
    json_extract_string(js,"omp_bind", cfg.omp_bind);
    json_extract_string(js,"omp_places", cfg.omp_places);

    json_extract_string(js,"subcsr_meta_dir", cfg.subcsr_meta_dir);
    json_extract_string(js,"subcsr_cluster_dir", cfg.subcsr_cluster_dir);

    json_extract_string(js,"sub_ids_path", cfg.sub_ids_path);
    json_extract_string(js,"sub_offsets_path", cfg.sub_offsets_path);
    json_extract_string(js,"sub_bucket_id_path", cfg.sub_bucket_id_path);
    json_extract_string(js,"sub_cluster_id_path", cfg.sub_cluster_id_path);
    json_extract_string(js,"bucket_sub_offsets_path", cfg.bucket_sub_offsets_path);
    json_extract_string(js,"sub_centroids_path", cfg.sub_centroids_path);

    json_extract_bool(js,"sub_cluster_u16", cfg.sub_cluster_u16);
    json_extract_num(js,"csr_N", cfg.csr_N);
    { int tmp=0; if(json_extract_num(js,"csr_nlist", tmp)) cfg.csr_nlist=(uint32_t)tmp; }
    { int tmp=0; if(json_extract_num(js,"nlist", tmp))     cfg.csr_nlist=(uint32_t)tmp; }
    json_extract_num(js,"sub_centroids_d", cfg.sub_centroids_d);
    json_extract_num(js,"sub_codes_d", cfg.sub_codes_d);

    json_extract_num(js,"max_score", cfg.max_score);

    json_extract_num(js,"warmup_enum_iters", cfg.warmup_enum_iters);
    json_extract_num(js,"warmup_q", cfg.warmup_q);
    json_extract_bool(js,"warmup_random", cfg.warmup_random);
    json_extract_bool(js,"warmup_use_real_logits", cfg.warmup_use_real_logits);
    json_extract_num(js,"random_seed", cfg.random_seed);

    json_extract_num(js,"centroid_heap_k", cfg.centroid_heap_k);
    json_extract_num(js,"centroid_kernel_batch_clusters", cfg.centroid_kernel_batch_clusters);
    json_extract_bool(js,"merge_adjacent_spans", cfg.merge_adjacent_spans);
    json_extract_bool(js,"sort_bucket_ids_for_centroid", cfg.sort_bucket_ids_for_centroid);
    json_extract_string(js,"centroid_batch_mode", cfg.centroid_batch_mode);

    json_extract_bool(js,"singleton_direct_refine", cfg.singleton_direct_refine);

    json_extract_bool(js,"centroid_early_stop", cfg.centroid_early_stop);
    json_extract_num(js,"centroid_early_stop_after_bins", cfg.centroid_early_stop_after_bins);
    json_extract_string(js,"centroid_gate_mode", cfg.centroid_gate_mode);
    json_extract_string(js,"calib_margin_csv", cfg.calib_margin_csv);
    json_extract_string(js,"calib_margin_quantile", cfg.calib_margin_quantile);
    json_extract_num(js,"calib_batch_qual_ratio_eps", cfg.calib_batch_qual_ratio_eps);
    json_extract_num(js,"calib_batch_qual_abs_eps", cfg.calib_batch_qual_abs_eps);
    json_extract_num(js,"calib_min_batches_before_stop", cfg.calib_min_batches_before_stop);
    json_extract_num(js,"heap_gate_by_calib_csv", cfg.heap_gate_by_calib_csv);

    json_extract_num(js,"calib_lowqual_use_heap_updates", cfg.calib_lowqual_use_heap_updates);
    json_extract_num(js,"calib_heap_front_alpha", cfg.calib_heap_front_alpha);
    json_extract_num(js,"calib_heap_front_update_ratio_eps", cfg.calib_heap_front_update_ratio_eps);
    json_extract_num(js,"calib_heap_front_update_abs_eps", cfg.calib_heap_front_update_abs_eps);
    json_extract_num(js,"calib_heap_updates_abs_eps", cfg.calib_heap_updates_abs_eps);

    if(cfg.sub_centroids_d <= 0) cfg.sub_centroids_d = cfg.D_logits;
    if(cfg.sub_codes_d <= 0) cfg.sub_codes_d = cfg.D_in;
}

static inline void override_cli(Config& cfg, int argc, char** argv){
    auto need=[&](int i){ if(i+1>=argc) die(std::string("missing value for ")+argv[i]); };
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--json"){ need(i); apply_json(cfg, slurp(argv[++i])); }
        else if(a=="--logits_path"){ need(i); cfg.logits_path=argv[++i]; }
        else if(a=="--thr_f32_path"){ need(i); cfg.thr_f32_path=argv[++i]; }
        else if(a=="--Q"){ need(i); cfg.Q=std::stoi(argv[++i]); }
        else if(a=="--D"){ need(i); cfg.D_logits=std::stoi(argv[++i]); cfg.sub_centroids_d=cfg.D_logits; }
        else if(a=="--D_in"){ need(i); cfg.D_in=std::stoi(argv[++i]); }
        else if(a=="--omp"){ need(i); cfg.omp_threads=std::stoi(argv[++i]); }
        else if(a=="--repeats"){ need(i); cfg.repeats=std::stoi(argv[++i]); }
        else if(a=="--warmup_enum_iters"){ need(i); cfg.warmup_enum_iters=std::stoi(argv[++i]); }
        else if(a=="--warmup_q"){ need(i); cfg.warmup_q=std::stoi(argv[++i]); }
        else if(a=="--warmup_use_real_logits"){ need(i); cfg.warmup_use_real_logits=(std::stoi(argv[++i])!=0); }
        else if(a=="--sched_enum"){ need(i); cfg.sched_enum=argv[++i]; }
        else if(a=="--chunk_enum"){ need(i); cfg.chunk_enum=std::stoi(argv[++i]); }

        else if(a=="--subcsr_meta_dir"){ need(i); cfg.subcsr_meta_dir=argv[++i]; }
        else if(a=="--subcsr_cluster_dir"){ need(i); cfg.subcsr_cluster_dir=argv[++i]; }

        else if(a=="--omp_bind"){ need(i); cfg.omp_bind=argv[++i]; }
        else if(a=="--omp_places"){ need(i); cfg.omp_places=argv[++i]; }

        else if(a=="--centroid_heap_k"){ need(i); cfg.centroid_heap_k=std::stoi(argv[++i]); }
        else if(a=="--centroid_kernel_batch_clusters"){ need(i); cfg.centroid_kernel_batch_clusters=std::stoi(argv[++i]); }
        else if(a=="--merge_adjacent_spans"){ need(i); cfg.merge_adjacent_spans=(std::stoi(argv[++i])!=0); }
        else if(a=="--sort_bucket_ids_for_centroid"){ need(i); cfg.sort_bucket_ids_for_centroid=(std::stoi(argv[++i])!=0); }
        else if(a=="--centroid_batch_mode"){ need(i); cfg.centroid_batch_mode=argv[++i]; }
        else if(a=="--max_score"){ need(i); cfg.max_score=(float)std::atof(argv[++i]); }

        else if(a=="--singleton_direct_refine"){ need(i); cfg.singleton_direct_refine=(std::stoi(argv[++i])!=0); }

        else if(a=="--centroid_gate_mode"){ need(i); cfg.centroid_gate_mode=argv[++i]; }
        else if(a=="--calib_margin_csv"){ need(i); cfg.calib_margin_csv=argv[++i]; }
        else if(a=="--calib_margin_quantile"){ need(i); cfg.calib_margin_quantile=argv[++i]; }
        else if(a=="--centroid_early_stop"){ need(i); cfg.centroid_early_stop=(std::stoi(argv[++i])!=0); }
        else if(a=="--centroid_early_stop_after_bins"){ need(i); cfg.centroid_early_stop_after_bins=std::stoi(argv[++i]); }
        else if(a=="--calib_batch_qual_ratio_eps"){ need(i); cfg.calib_batch_qual_ratio_eps=(float)std::atof(argv[++i]); }
        else if(a=="--calib_batch_qual_abs_eps"){ need(i); cfg.calib_batch_qual_abs_eps=std::stoi(argv[++i]); }
        else if(a=="--calib_min_batches_before_stop"){ need(i); cfg.calib_min_batches_before_stop=std::stoi(argv[++i]); }
        else if(a=="--heap_gate_by_calib_csv"){ need(i); cfg.heap_gate_by_calib_csv=(std::stoi(argv[++i])!=0); }

        else if(a=="--calib_lowqual_use_heap_updates"){ need(i); cfg.calib_lowqual_use_heap_updates=std::stoi(argv[++i]); }
        else if(a=="--calib_heap_front_alpha"){ need(i); cfg.calib_heap_front_alpha=(float)std::atof(argv[++i]); }
        else if(a=="--calib_heap_front_update_ratio_eps"){ need(i); cfg.calib_heap_front_update_ratio_eps=(float)std::atof(argv[++i]); }
        else if(a=="--calib_heap_front_update_abs_eps"){ need(i); cfg.calib_heap_front_update_abs_eps=std::stoi(argv[++i]); }
        else if(a=="--calib_heap_updates_abs_eps"){ need(i); cfg.calib_heap_updates_abs_eps=std::stoi(argv[++i]); }
    }
}

// ---------------- misc infer / random ----------------
static inline uint64_t infer_nlist_from_bucket_sub_offsets(const std::string& p){
    struct stat st;
    if (::stat(p.c_str(), &st)!=0) die("cannot stat bucket_sub_offsets: "+p);
    if ((st.st_size % 8)!=0) die("bucket_sub_offsets size not multiple of 8: "+p);
    uint64_t len=(uint64_t)st.st_size/8;
    if(len<2) die("bucket_sub_offsets too small (<2): "+p);
    return len-1;
}
static inline void fill_random_logits(std::vector<float>& out, int Q, int D, int seed){
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f,1.f);
    out.resize((size_t)Q*(size_t)D);
    for(auto& x: out) x=nd(rng);
}

// ---------------- spans / kernels ----------------
struct SpanU64 { uint64_t s=0, e=0; };

static inline void merge_spans_inplace(std::vector<SpanU64>& spans, bool merge_adjacent, int batch_cap){
    if(spans.empty()) return;
    std::sort(spans.begin(), spans.end(), [](const SpanU64&a,const SpanU64&b){
        return (a.s<b.s) || (a.s==b.s && a.e<b.e);
    });
    std::vector<SpanU64> out;
    out.reserve(spans.size());
    SpanU64 cur = spans[0];
    for(size_t i=1;i<spans.size();++i){
        const auto &x = spans[i];
        bool contiguous = (merge_adjacent ? (x.s <= cur.e) : (x.s < cur.e));
        bool cap_ok = (batch_cap<=0) ? true : ((int)((std::max(cur.e, x.e) - cur.s)) <= batch_cap);
        if(contiguous && cap_ok){
            if(x.e > cur.e) cur.e = x.e;
        } else {
            out.push_back(cur);
            cur = x;
        }
    }
    out.push_back(cur);
    spans.swap(out);
}

static inline void compute_l2sq_batch_f32(
    const float* q_ptr,
    const float* rows_ptr,
    int nrows,
    int d,
    float* out_dist)
{
    for(int i=0;i<nrows;i++){
        const float* x = rows_ptr + (size_t)i*(size_t)d;
        float s = 0.0f;
        int j = 0;
        for(; j + 7 < d; j += 8){
            float a0=q_ptr[j+0]-x[j+0]; s += a0*a0;
            float a1=q_ptr[j+1]-x[j+1]; s += a1*a1;
            float a2=q_ptr[j+2]-x[j+2]; s += a2*a2;
            float a3=q_ptr[j+3]-x[j+3]; s += a3*a3;
            float a4=q_ptr[j+4]-x[j+4]; s += a4*a4;
            float a5=q_ptr[j+5]-x[j+5]; s += a5*a5;
            float a6=q_ptr[j+6]-x[j+6]; s += a6*a6;
            float a7=q_ptr[j+7]-x[j+7]; s += a7*a7;
        }
        for(; j < d; ++j){
            float a = q_ptr[j] - x[j];
            s += a*a;
        }
        out_dist[i] = s;
    }
}

// ---------- singleton split ----------
struct BucketSplitStats {
    uint64_t singleton_bucket_hits = 0;
    uint64_t singleton_clusters_direct = 0;
    uint64_t nonsingleton_clusters_centroid_path = 0;
};

static inline BucketSplitStats split_bins_singleton_vs_nonsingleton(
    const CSRSubCSRIndexRuntime& subidx,
    const std::vector<std::vector<uint32_t>>& bins_buckets_in,
    std::vector<std::vector<uint32_t>>& bins_buckets_nonsingleton_out,
    std::vector<uint32_t>& direct_refine_clusters_out,
    bool enable_singleton_direct_refine)
{
    BucketSplitStats st{};
    const int NB = (int)bins_buckets_in.size();
    if((int)bins_buckets_nonsingleton_out.size() != NB){
        bins_buckets_nonsingleton_out.assign((size_t)NB, {});
    }

    for(int b=0; b<NB; ++b){
        auto& dst = bins_buckets_nonsingleton_out[(size_t)b];
        dst.clear();

        const auto& src = bins_buckets_in[(size_t)b];
        if(src.empty()) continue;

        dst.reserve(src.size());

        for(uint32_t bid : src){
            if((size_t)bid >= (size_t)subidx.nlist) continue;

            const uint64_t s = subidx.bucket_sub_offsets[(size_t)bid];
            const uint64_t e = subidx.bucket_sub_offsets[(size_t)bid + 1];
            const uint64_t ncl = (e > s ? (e - s) : 0);

            if(ncl == 0){
                continue;
            }else if(ncl == 1 && enable_singleton_direct_refine){
                st.singleton_bucket_hits += 1;
                st.singleton_clusters_direct += 1;
                if(s <= (uint64_t)std::numeric_limits<uint32_t>::max()){
                    direct_refine_clusters_out.push_back((uint32_t)s);
                }else{
                    die("singleton sub-centroid id exceeds uint32 range");
                }
            }else{
                dst.push_back(bid);
                st.nonsingleton_clusters_centroid_path += ncl;
            }
        }
    }

    return st;
}

// ---------------- centroid stage ----------------
struct CentroidStageResult {
    uint64_t clusters_seen = 0;
    uint64_t kernel_calls = 0;
    uint64_t gather_clusters = 0;
    uint64_t spans_before = 0;
    uint64_t spans_after = 0;

    double gather_ms = 0.0;
    double batch_build_ms = 0.0;
    double kernel_ms = 0.0;
    double sink_ms = 0.0;
    double expand_ms = 0.0;

    bool early_stopped = false;
    int bins_processed = 0;

    bool heap_full_at_end = false;
    float best_seen_final = std::numeric_limits<float>::infinity();
    float heap_worst_final = std::numeric_limits<float>::infinity();
    int consecutive_low_qual_final = 0;

    // compact counters kept for integration sanity-check
    uint64_t heap_push_attempts = 0;
    uint64_t heap_updates = 0;
    uint64_t heap_rejects = 0;
    uint64_t heap_appends = 0;
    uint64_t heap_replaces = 0;
};

struct CalibCsvStopDiag {
    uint64_t total_batches = 0;
    uint64_t low_qual_batches = 0;
    uint64_t heap_updates_total = 0;
    uint64_t heap_front_updates_total = 0;

    float final_margin_from_csv = 0.0f;
    float final_batch_front_ratio = 0.0f;
};

static inline void append_bucket_cluster_spans_from_bin(
    const CSRSubCSRIndexRuntime& subidx,
    const std::vector<uint32_t>& bucket_ids_in_bin,
    std::vector<SpanU64>& spans_out)
{
    for(uint32_t bid : bucket_ids_in_bin){
        if ((size_t)bid >= (size_t)subidx.nlist) continue;
        uint64_t s = subidx.bucket_sub_offsets[(size_t)bid];
        uint64_t e = subidx.bucket_sub_offsets[(size_t)bid + 1];
        if(e > s) spans_out.push_back({s,e});
    }
}

static inline bool calib_csv_should_stop_after_batch(
    const Config& cfg,
    int bins_processed,
    bool heap_full,
    int consecutive_low_qual_batches)
{
    if(!cfg.centroid_early_stop) return false;
    if(!heap_full) return false;
    if(bins_processed < std::max(0, cfg.centroid_early_stop_after_bins)) return false;
    if(consecutive_low_qual_batches < std::max(1, cfg.calib_min_batches_before_stop)) return false;
    return true;
}

static inline CentroidStageResult run_centroid_stage_copy_calibcsv_one_query(
    const Config& cfg,
    const CalibMarginTable& calib_tab,
    const CSRSubCSRIndexRuntime& subidx,
    const float* q_logits,
    const std::vector<std::vector<uint32_t>>& bins_buckets, // non-singleton only
    TopKHeap& heap,
    float* tmp_batch_f32,
    float* tmp_dist_f32,
    std::vector<SpanU64>& spans_tmp,
    std::vector<uint32_t>* tmp_cids_buf,
    CalibCsvStopDiag* diag_out /*optional*/)
{
    if (!tmp_cids_buf) die("tmp_cids_buf must not be null");

    CentroidStageResult rr;
    CalibCsvStopDiag diag_local;

    const int D = cfg.sub_centroids_d;
    const int batch_cap = std::max(1, cfg.centroid_kernel_batch_clusters);

    uint64_t clusters_seen = 0;
    float best_seen = std::numeric_limits<float>::infinity();
    int bins_processed = 0;
    int consecutive_low_qual_batches = 0;

    for(int b = 0; b < (int)bins_buckets.size(); ++b){
        const auto& bucket_ids = bins_buckets[(size_t)b];
        if(bucket_ids.empty()) continue;

        spans_tmp.clear();
        auto t_expand0 = WallClock::now();
        append_bucket_cluster_spans_from_bin(subidx, bucket_ids, spans_tmp);
        rr.expand_ms += ms_since(t_expand0);
        rr.spans_before += (uint64_t)spans_tmp.size();

        if(spans_tmp.empty()){
            bins_processed++;
            rr.bins_processed = bins_processed;
            continue;
        }

        if(cfg.sort_bucket_ids_for_centroid){
            std::sort(spans_tmp.begin(), spans_tmp.end(), [](const SpanU64& a, const SpanU64& b){
                return (a.s < b.s) || (a.s == b.s && a.e < b.e);
            });
        }

        if(cfg.merge_adjacent_spans){
            merge_spans_inplace(spans_tmp, /*merge_adjacent=*/true, /*batch_cap=*/0);
        }else{
            std::sort(spans_tmp.begin(), spans_tmp.end(), [](const SpanU64& a, const SpanU64& b){
                return (a.s < b.s) || (a.s == b.s && a.e < b.e);
            });
        }
        rr.spans_after += (uint64_t)spans_tmp.size();

        size_t sp_i = 0;
        while(sp_i < spans_tmp.size()){
            auto t_build0 = WallClock::now();

            int n_this = 0;
            size_t sp_j = sp_i;

            tmp_cids_buf->clear();
            tmp_cids_buf->reserve((size_t)batch_cap);

            for(; sp_j < spans_tmp.size(); ++sp_j){
                const auto& sp = spans_tmp[sp_j];
                uint64_t len64 = (sp.e > sp.s ? (sp.e - sp.s) : 0);
                if(len64 == 0) continue;
                if(len64 > (uint64_t)INT32_MAX) die("span too large");
                int len = (int)len64;

                if(n_this > 0 && n_this + len > batch_cap) break;
                if(n_this == 0 && len > batch_cap){
                    len = batch_cap;
                }

                const float* src = subidx.sub_centroids + (size_t)sp.s * (size_t)D;
                float* dst = tmp_batch_f32 + (size_t)n_this * (size_t)D;
                std::memcpy(dst, src, (size_t)len * (size_t)D * sizeof(float));

                // FIX: store REAL sub-centroid IDs aligned with copied rows
                for(int t = 0; t < len; ++t){
                    uint64_t cid64 = sp.s + (uint64_t)t;
                    if(cid64 > (uint64_t)std::numeric_limits<uint32_t>::max()){
                        die("sub-centroid id exceeds uint32 range");
                    }
                    tmp_cids_buf->push_back((uint32_t)cid64);
                }

                n_this += len;
                rr.gather_clusters += (uint64_t)len;

                if((int)len64 > len){
                    spans_tmp[sp_j].s = sp.s + (uint64_t)len;
                    break;
                }
            }

            const double gms = ms_since(t_build0);
            rr.gather_ms += gms;
            rr.batch_build_ms += gms;

            if(n_this <= 0){
                sp_i = sp_j + 1;
                continue;
            }
            if((int)tmp_cids_buf->size() != n_this) die("tmp_cids_buf size mismatch with n_this");

            if(sp_j < spans_tmp.size()){
                if(spans_tmp[sp_j].s < spans_tmp[sp_j].e) sp_i = sp_j;
                else sp_i = sp_j + 1;
            }else{
                sp_i = sp_j;
            }

            auto t_k0 = WallClock::now();
            compute_l2sq_batch_f32(q_logits, tmp_batch_f32, n_this, D, tmp_dist_f32);
            rr.kernel_ms += ms_since(t_k0);
            rr.kernel_calls++;
            rr.clusters_seen += (uint64_t)n_this;
            clusters_seen += (uint64_t)n_this;

            auto t_s0 = WallClock::now();

            // pass1: best_seen
            for(int i = 0; i < n_this; ++i){
                const float d2 = tmp_dist_f32[i];
                if(d2 < best_seen) best_seen = d2;
            }

            float margin_csv = calib_tab.margin_from_best2(best_seen);
            if(!(margin_csv >= 0.0f) || !std::isfinite(margin_csv)) margin_csv = 0.0f;
            const float thr = best_seen + margin_csv;

            float front_alpha = cfg.calib_heap_front_alpha;
            if(!std::isfinite(front_alpha) || front_alpha < 0.f) front_alpha = 0.f;
            const float front_thr = best_seen + front_alpha * margin_csv;

            uint64_t batch_heap_updates = 0;
            uint64_t batch_heap_front_updates = 0;
            uint64_t batch_qual_abs = 0;

            for(int i = 0; i < n_this; ++i){
                const float d2 = tmp_dist_f32[i];
                const bool under_curve = (d2 <= thr);
                if(under_curve) batch_qual_abs++;

                if(cfg.heap_gate_by_calib_csv && !under_curve){
                    continue;
                }

                rr.heap_push_attempts++;

                // FIXED: real id, not fake placeholder id
                const uint32_t real_cid = (*tmp_cids_buf)[(size_t)i];
                HeapPushOutcome outc = heap.push_with_outcome(d2, real_cid);

                if(outc == HeapPushOutcome::Rejected){
                    rr.heap_rejects++;
                }else{
                    batch_heap_updates++;
                    rr.heap_updates++;
                    if(outc == HeapPushOutcome::Appended) rr.heap_appends++;
                    else if(outc == HeapPushOutcome::Replaced) rr.heap_replaces++;

                    if(d2 <= front_thr) batch_heap_front_updates++;
                }
            }

            const float batch_qual_ratio = (n_this > 0)
                ? (float)((double)batch_qual_abs / (double)n_this) : 0.0f;
            (void)batch_qual_ratio; // only used in fallback branch below

            const float batch_heap_front_ratio = (batch_heap_updates > 0)
                ? (float)((double)batch_heap_front_updates / (double)batch_heap_updates) : 0.0f;

            diag_local.total_batches++;
            diag_local.heap_updates_total += batch_heap_updates;
            diag_local.heap_front_updates_total += batch_heap_front_updates;
            diag_local.final_margin_from_csv = margin_csv;
            diag_local.final_batch_front_ratio = batch_heap_front_ratio;

            bool low_qual = false;

            if(cfg.calib_lowqual_use_heap_updates){
                const bool low_updates_zero = (batch_heap_updates == 0);

                bool low_updates_abs = false;
                if(cfg.calib_heap_updates_abs_eps > 0){
                    low_updates_abs = ((int)batch_heap_updates <= cfg.calib_heap_updates_abs_eps);
                }

                bool low_front_ratio = false;
                if(cfg.calib_heap_front_update_ratio_eps > 0.0f && batch_heap_updates > 0){
                    low_front_ratio = (batch_heap_front_ratio <= cfg.calib_heap_front_update_ratio_eps);
                }

                bool low_front_abs = false;
                if(cfg.calib_heap_front_update_abs_eps > 0){
                    low_front_abs = ((int)batch_heap_front_updates <= cfg.calib_heap_front_update_abs_eps);
                }

                if(low_updates_zero){
                    low_qual = true;
                }else{
                    bool any_secondary = false;
                    bool sec_low = false;
                    if(cfg.calib_heap_updates_abs_eps > 0){ any_secondary = true; sec_low = sec_low || low_updates_abs; }
                    if(cfg.calib_heap_front_update_ratio_eps > 0.0f){ any_secondary = true; sec_low = sec_low || low_front_ratio; }
                    if(cfg.calib_heap_front_update_abs_eps > 0){ any_secondary = true; sec_low = sec_low || low_front_abs; }
                    low_qual = any_secondary ? sec_low : false;
                }
            } else {
                const bool low_by_ratio = (cfg.calib_batch_qual_ratio_eps > 0.0f &&
                                           batch_qual_ratio <= cfg.calib_batch_qual_ratio_eps);
                const bool low_by_abs   = (cfg.calib_batch_qual_abs_eps > 0 &&
                                           (int)batch_qual_abs <= cfg.calib_batch_qual_abs_eps);

                if(cfg.calib_batch_qual_ratio_eps > 0.0f && cfg.calib_batch_qual_abs_eps > 0){
                    low_qual = (low_by_ratio || low_by_abs);
                }else if(cfg.calib_batch_qual_ratio_eps > 0.0f){
                    low_qual = low_by_ratio;
                }else if(cfg.calib_batch_qual_abs_eps > 0){
                    low_qual = low_by_abs;
                }else{
                    low_qual = (batch_qual_abs == 0);
                }
            }

            if(low_qual){
                consecutive_low_qual_batches++;
                diag_local.low_qual_batches++;
            }else{
                consecutive_low_qual_batches = 0;
            }

            rr.sink_ms += ms_since(t_s0);

            const bool stop = calib_csv_should_stop_after_batch(
                cfg,
                /*bins_processed=*/bins_processed + 1,
                /*heap_full=*/heap.full(),
                /*consecutive_low_qual_batches=*/consecutive_low_qual_batches
            );

            if(stop){
                rr.early_stopped = true;
                rr.bins_processed = bins_processed + 1;
                rr.heap_full_at_end = heap.full();
                rr.best_seen_final = best_seen;
                rr.heap_worst_final = heap.worst();
                rr.consecutive_low_qual_final = consecutive_low_qual_batches;
                rr.clusters_seen = clusters_seen;
                if(diag_out) *diag_out = diag_local;
                return rr;
            }
        }

        bins_processed++;
        rr.bins_processed = bins_processed;
    }

    rr.heap_full_at_end = heap.full();
    rr.best_seen_final = best_seen;
    rr.heap_worst_final = heap.worst();
    rr.consecutive_low_qual_final = consecutive_low_qual_batches;
    rr.clusters_seen = clusters_seen;
    if(diag_out) *diag_out = diag_local;
    return rr;
}

// ---------------- round stats (clean) ----------------
struct RoundStats {
    double t_load_sub_ms=0, t_load_logits_ms=0, t_prepare_ms=0, t_enum_stage_ms=0, t_total_ms=0, warmup_ms=0;
    double bins_build_ms_sum=0, centroid_ms_sum=0;
    double centroid_expand_ms_sum=0, centroid_kernel_ms_sum=0, centroid_sink_ms_sum=0, centroid_gather_ms_sum=0;

    uint64_t centroid_kernel_calls=0, centroid_kernel_clusters_total=0;
    uint64_t buckets_enum_total=0, clusters_enum_total=0, bins_nonempty_total=0;

    uint64_t centroid_spans_before_total = 0;
    uint64_t centroid_spans_after_total  = 0;

    // singleton/direct-refine
    uint64_t singleton_bucket_hits_total = 0;
    uint64_t singleton_clusters_direct_refine_total = 0;
    uint64_t nonsingleton_clusters_centroid_path_total = 0;
    uint64_t direct_refine_clusters_merged_total = 0;
    uint64_t centroid_selected_clusters_merged_total = 0;
    uint64_t merged_refine_candidates_total = 0;

    // heap stats
    uint64_t heap_selected_total = 0;
    uint64_t heap_push_attempts_total = 0;
    uint64_t heap_updates_total = 0;
    uint64_t heap_rejects_total = 0;
    uint64_t heap_appends_total = 0;
    uint64_t heap_replaces_total = 0;
    uint64_t heap_full_q = 0;

    // early stop
    uint64_t early_stop_q = 0;
    uint64_t early_stop_bins_skipped = 0;

    // compact calib summary
    uint64_t calib_q_eval = 0;
    uint64_t calib_q_triggered = 0;
    uint64_t calib_q_heap_full_end = 0;
    uint64_t calib_q_heap_not_full_end = 0;
    uint64_t calib_total_batches = 0;
    uint64_t calib_low_qual_batches = 0;
};

static inline void print_round_summary(const Config& cfg, const RoundStats& s, int rep){
    const double enum_qps  = (s.t_enum_stage_ms > 0.0) ? (1000.0 * (double)cfg.Q / s.t_enum_stage_ms) : 0.0;
    const double total_qps = (s.t_total_ms > 0.0)      ? (1000.0 * (double)cfg.Q / s.t_total_ms)      : 0.0;

    std::cout << "[REP] rep=" << rep
              << " prepare_ms=" << s.t_prepare_ms
              << " enum_stage_ms=" << s.t_enum_stage_ms
              << " total_ms=" << s.t_total_ms
              << " enum_qps=" << enum_qps
              << " total_qps=" << total_qps
              << "\n";

    std::cout << "[WORK]"
              << " buckets_enum_total=" << s.buckets_enum_total
              << " clusters_enum_total=" << s.clusters_enum_total
              << " heap_selected_total=" << s.heap_selected_total
              << " heap_push_attempts_total=" << s.heap_push_attempts_total
              << " heap_updates_total=" << s.heap_updates_total
              << " heap_rejects_total=" << s.heap_rejects_total
              << " early_stop_q=" << s.early_stop_q
              << " early_stop_bins_skipped=" << s.early_stop_bins_skipped
              << "\n";

    std::cout << "[SINGLETON_SPLIT]"
              << " singleton_bucket_hits_total=" << s.singleton_bucket_hits_total
              << " singleton_clusters_direct_refine_total=" << s.singleton_clusters_direct_refine_total
              << " nonsingleton_clusters_centroid_path_total=" << s.nonsingleton_clusters_centroid_path_total
              << " merged_refine_candidates_total=" << s.merged_refine_candidates_total
              << "\n";

    std::cout << "[CENTROID.BATCH]"
              << " gather_ms_sum=" << s.centroid_gather_ms_sum
              << " kernel_ms_sum=" << s.centroid_kernel_ms_sum
              << " kernel_calls=" << s.centroid_kernel_calls
              << " kernel_clusters_total=" << s.centroid_kernel_clusters_total
              << " spans_before_total=" << s.centroid_spans_before_total
              << " spans_after_total=" << s.centroid_spans_after_total
              << "\n";

    if (s.calib_q_eval > 0) {
        const double trigger_frac =
            (double)s.calib_q_triggered / (double)std::max<uint64_t>(1, s.calib_q_eval);
        const double lowqual_batch_ratio =
            (double)s.calib_low_qual_batches / (double)std::max<uint64_t>(1, s.calib_total_batches);

        std::cout << "[EARLY_STOP.CALIBCSV]"
                  << " enabled=" << (cfg.centroid_early_stop ? 1 : 0)
                  << " q_eval=" << s.calib_q_eval
                  << " q_triggered=" << s.calib_q_triggered
                  << " trigger_frac=" << trigger_frac
                  << " q_heap_full_end=" << s.calib_q_heap_full_end
                  << " q_heap_not_full_end=" << s.calib_q_heap_not_full_end
                  << " total_batches=" << s.calib_total_batches
                  << " low_qual_batches=" << s.calib_low_qual_batches
                  << " low_qual_batch_ratio=" << lowqual_batch_ratio
                  << "\n";
    }
}

// ---------------- run_round ----------------
static inline RoundStats run_round(Config cfg){
    RoundStats S;
    auto t_total0 = WallClock::now();

    if(cfg.Q <= 0) die("Q must be > 0");
    if(cfg.D_logits <= 0) die("D_logits must be > 0");
    if(cfg.sub_centroids_d <= 0) die("sub_centroids_d must be > 0");
    if(cfg.sub_centroids_d != cfg.D_logits){
        std::cerr << "[WARN] sub_centroids_d (" << cfg.sub_centroids_d
                  << ") != D_logits (" << cfg.D_logits
                  << "), forcing sub_centroids_d=D_logits\n";
        cfg.sub_centroids_d = cfg.D_logits;
    }

    if(cfg.centroid_batch_mode.empty()) cfg.centroid_batch_mode = "copy";
    if(cfg.centroid_batch_mode != "copy"){
        std::cerr << "[WARN] forcing centroid_batch_mode=copy (only copy supported)\n";
        cfg.centroid_batch_mode = "copy";
    }

    if(cfg.centroid_gate_mode != "calib_csv"){
        die("This file only supports --centroid_gate_mode calib_csv");
    }
    if(cfg.calib_margin_csv.empty()){
        die("calib_csv mode requires --calib_margin_csv");
    }
    if(cfg.calib_margin_quantile != "q95" && cfg.calib_margin_quantile != "q99"){
        std::cerr << "[WARN] invalid calib_margin_quantile=" << cfg.calib_margin_quantile
                  << ", fallback to q99\n";
        cfg.calib_margin_quantile = "q99";
    }

    CalibMarginTable calib_tab = load_calib_margin_csv_or_die(cfg.calib_margin_csv, cfg.calib_margin_quantile);

    set_env_if_nonempty("OMP_PROC_BIND", cfg.omp_bind);
    set_env_if_nonempty("OMP_PLACES", cfg.omp_places);

#ifdef _OPENMP
    if(cfg.omp_threads > 0){
        omp_set_dynamic(0);
        omp_set_num_threads(cfg.omp_threads);
    }
    omp_apply_schedule(cfg.sched_enum, cfg.chunk_enum);
#endif

    if(cfg.subcsr_meta_dir.empty())    die("subcsr_meta_dir required");
    if(cfg.subcsr_cluster_dir.empty()) die("subcsr_cluster_dir required");
    if(cfg.thr_f32_path.empty())       die("thr_f32_path required");
    if(cfg.logits_path.empty())        die("logits_path required");

    const std::string p_sub_offsets    = pick_path_or_default(cfg.sub_offsets_path,       cfg.subcsr_meta_dir, "sub_offsets.u64.bin");
    const std::string p_sub_ids        = pick_path_or_default(cfg.sub_ids_path,           cfg.subcsr_meta_dir, "sub_ids.i64.csr.bin");
    const std::string p_sub_bucket     = pick_path_or_default(cfg.sub_bucket_id_path,     cfg.subcsr_meta_dir, "sub_bucket_id.u32.bin");
    const std::string p_sub_cluster    = pick_path_or_default(cfg.sub_cluster_id_path,    cfg.subcsr_meta_dir, "sub_cluster_id.u32.bin");
    const std::string p_sub_cents      = pick_path_or_default(cfg.sub_centroids_path,     cfg.subcsr_meta_dir, "sub_centroids.f32.bin");
    const std::string p_bucket_sub_off = pick_path_or_default(cfg.bucket_sub_offsets_path,cfg.subcsr_meta_dir, "bucket_sub_offsets.u64.bin");

    const uint64_t nlist_inferred = infer_nlist_from_bucket_sub_offsets(p_bucket_sub_off);
    if(cfg.csr_nlist == 0) cfg.csr_nlist = (uint32_t)nlist_inferred;
    if((uint64_t)cfg.csr_nlist != nlist_inferred){
        std::cerr << "[WARN] cfg.csr_nlist=" << cfg.csr_nlist
                  << " mismatches inferred=" << nlist_inferred << ", overriding\n";
        cfg.csr_nlist = (uint32_t)nlist_inferred;
    }

    ensure_dir_fs(cfg.subcsr_cluster_dir);
    ensure_symlink_or_copy(p_sub_offsets,    path_join(cfg.subcsr_cluster_dir, "sub_offsets.u64.bin"));
    ensure_symlink_or_copy(p_sub_ids,        path_join(cfg.subcsr_cluster_dir, "sub_ids.i64.csr.bin"));
    ensure_symlink_or_copy(p_sub_bucket,     path_join(cfg.subcsr_cluster_dir, "sub_bucket_id.u32.bin"));
    ensure_symlink_or_copy(p_sub_cluster,    path_join(cfg.subcsr_cluster_dir, "sub_cluster_id.u32.bin"));
    ensure_symlink_or_copy(p_sub_cents,      path_join(cfg.subcsr_cluster_dir, "sub_centroids.f32.bin"));
    ensure_symlink_or_copy(p_bucket_sub_off, path_join(cfg.subcsr_cluster_dir, "bucket_sub_offsets.u64.bin"));

    auto t0 = WallClock::now();

    CSRSubCSRIndexRuntime subidx;
    subidx.load_subcsr(cfg.subcsr_cluster_dir, cfg.sub_centroids_d, cfg.csr_nlist);
    if(!subidx.sub_centroids) die("sub_centroids is empty after load");
    if(!subidx.bucket_sub_offsets || subidx.nlist == 0) die("bucket_sub_offsets invalid after load");
    S.t_load_sub_ms = ms_since(t0);

    t0 = WallClock::now();
    std::vector<float> logits = read_bin_or_die<float>(cfg.logits_path, (size_t)cfg.Q * (size_t)cfg.D_logits);
    S.t_load_logits_ms = ms_since(t0);

    std::vector<float> thr = read_bin_or_die<float>(cfg.thr_f32_path, (size_t)cfg.D_logits);

    auto plan = qbpg_bins_only_v3a::build_plan(cfg.Lsel, (int)cfg.Rmax);
    const int Lsel_eff = std::max(1, std::min(cfg.Lsel, cfg.D_logits));

    std::vector<uint32_t> base_ids((size_t)cfg.Q);
    std::vector<float> cost_rank((size_t)cfg.Q * (size_t)Lsel_eff);
    std::vector<uint32_t> rank2bit((size_t)cfg.Q * (size_t)Lsel_eff);

    // warmup enum prepare
    if(cfg.warmup_enum_iters > 0 && cfg.warmup_q > 0){
        const int warmQ = std::min(cfg.Q, cfg.warmup_q);
        std::vector<float> warm_logits;
        if(cfg.warmup_use_real_logits){
            warm_logits.assign(logits.begin(), logits.begin() + (size_t)warmQ * (size_t)cfg.D_logits);
        }else if(cfg.warmup_random){
            fill_random_logits(warm_logits, warmQ, cfg.D_logits, cfg.random_seed);
        }

        if(!warm_logits.empty()){
            auto tw0 = WallClock::now();
            for(int it=0; it<cfg.warmup_enum_iters; ++it){
#ifdef _OPENMP
#pragma omp parallel
#endif
                {
                    std::vector<float> cr; cr.reserve((size_t)Lsel_eff);
                    std::vector<uint32_t> r2b; r2b.reserve((size_t)Lsel_eff);
#ifdef _OPENMP
#pragma omp for schedule(runtime)
#endif
                    for(int qi=0; qi<warmQ; ++qi){
                        const float* lg = warm_logits.data() + (size_t)qi * (size_t)cfg.D_logits;
                        uint32_t base_id = 0;
                        cr.clear(); r2b.clear();
                        qbpg_v1::prepare_inputs_from_logits_thr_absdiff(
                            lg, thr.data(), cfg.D_logits, Lsel_eff, base_id, cr, r2b);
                    }
                }
            }
            S.warmup_ms = ms_since(tw0);
        }
    }

    // prepare all queries
    t0 = WallClock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> cr; cr.reserve((size_t)Lsel_eff);
        std::vector<uint32_t> r2b; r2b.reserve((size_t)Lsel_eff);
#ifdef _OPENMP
#pragma omp for schedule(runtime)
#endif
        for(int qi=0; qi<cfg.Q; ++qi){
            const float* lg = logits.data() + (size_t)qi * (size_t)cfg.D_logits;
            uint32_t base_id = 0;
            cr.clear(); r2b.clear();
            qbpg_v1::prepare_inputs_from_logits_thr_absdiff(
                lg, thr.data(), cfg.D_logits, Lsel_eff, base_id, cr, r2b);

            base_ids[(size_t)qi] = base_id;
            std::memcpy(cost_rank.data() + (size_t)qi * (size_t)Lsel_eff, cr.data(),  (size_t)Lsel_eff * sizeof(float));
            std::memcpy(rank2bit.data() + (size_t)qi * (size_t)Lsel_eff, r2b.data(), (size_t)Lsel_eff * sizeof(uint32_t));
        }
    }
    S.t_prepare_ms = ms_since(t0);

    qbpg_v1::EnumFilters filt;
    filt.keep_base = cfg.keep_base;
    filt.max_score = cfg.max_score;

    qbpg_bins_only_v3a::AlphaBinner binner;
    binner.NBINS = cfg.NBINS;
    binner.max_score = cfg.max_score;
    binner.alpha = 1.0f;

    t0 = WallClock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        double bins_build_ms_sum_t=0.0, centroid_ms_sum_t=0.0;
        double centroid_expand_ms_sum_t=0.0, centroid_gather_ms_sum_t=0.0, centroid_kernel_ms_sum_t=0.0, centroid_sink_ms_sum_t=0.0;

        uint64_t centroid_kernel_calls_t=0, centroid_kernel_clusters_total_t=0;
        uint64_t bins_nonempty_total_t=0, buckets_enum_total_t=0, clusters_enum_total_t=0;

        uint64_t singleton_bucket_hits_total_t = 0;
        uint64_t singleton_clusters_direct_refine_total_t = 0;
        uint64_t nonsingleton_clusters_centroid_path_total_t = 0;
        uint64_t direct_refine_clusters_merged_total_t = 0;
        uint64_t centroid_selected_clusters_merged_total_t = 0;
        uint64_t merged_refine_candidates_total_t = 0;

        uint64_t heap_selected_total_t=0, heap_push_attempts_total_t=0, heap_updates_total_t=0, heap_rejects_total_t=0;
        uint64_t heap_appends_total_t=0, heap_replaces_total_t=0, heap_full_q_t=0;

        uint64_t early_stop_q_t = 0, early_stop_bins_skipped_t = 0;

        uint64_t calib_q_eval_t = 0, calib_q_triggered_t = 0, calib_q_heap_full_end_t = 0, calib_q_heap_not_full_end_t = 0;
        uint64_t calib_total_batches_t = 0, calib_low_qual_batches_t = 0;

        uint64_t centroid_spans_before_total_t=0, centroid_spans_after_total_t=0;

        qbpg_v1::QueryBins local; local.init(cfg.NBINS);
        std::vector<float> ls; std::vector<uint32_t> lm;
        std::vector<std::vector<uint32_t>> bins_buckets((size_t)cfg.NBINS);
        std::vector<std::vector<uint32_t>> bins_buckets_nonsingleton((size_t)cfg.NBINS);

        std::vector<uint32_t> direct_refine_clusters;
        std::vector<uint32_t> centroid_selected_clusters;
        std::vector<uint32_t> merged_refine_clusters;
        direct_refine_clusters.reserve(4096);
        centroid_selected_clusters.reserve(4096);
        merged_refine_clusters.reserve(8192);

        TopKHeap heap; heap.reset(cfg.centroid_heap_k);
        std::vector<SpanU64> spans_tmp; spans_tmp.reserve(1024);

        const int batch_cap = std::max(1, cfg.centroid_kernel_batch_clusters);
        std::vector<float> tmp_batch_f32((size_t)batch_cap * (size_t)cfg.sub_centroids_d);
        std::vector<float> tmp_dist_f32((size_t)batch_cap);
        std::vector<uint32_t> tmp_cids_buf; tmp_cids_buf.reserve((size_t)batch_cap);

#ifdef _OPENMP
#pragma omp for schedule(runtime)
#endif
        for(int qi=0; qi<cfg.Q; ++qi){
            local.ids.clear();
            if((int)local.offsets.size() == cfg.NBINS + 1) std::fill(local.offsets.begin(), local.offsets.end(), 0u);
            else local.init(cfg.NBINS);
            ls.clear(); lm.clear();

            auto tb0 = WallClock::now();
            qbpg_bins_only_v3a::build_bins_full_dp(
                plan,
                base_ids[(size_t)qi],
                cost_rank.data() + (size_t)qi * (size_t)Lsel_eff,
                rank2bit.data() + (size_t)qi * (size_t)Lsel_eff,
                filt, cfg.NBINS, binner, cfg.cap_per_bin,
                ls, lm, local, nullptr, nullptr, 0);
            bins_build_ms_sum_t += ms_since(tb0);

            for(int b=0; b<cfg.NBINS; ++b) bins_buckets[(size_t)b].clear();

            for(int b=0; b<cfg.NBINS; ++b){
                const uint32_t s0 = local.offsets[(size_t)b];
                const uint32_t s1 = local.offsets[(size_t)b+1];
                const uint32_t len = s1 - s0;

                if(len){
                    ++bins_nonempty_total_t;
                    bins_buckets[(size_t)b].reserve((size_t)len);
                    for(uint32_t k=s0; k<s1; ++k){
                        bins_buckets[(size_t)b].push_back(local.ids[(size_t)k]);
                    }
                }
            }

            buckets_enum_total_t += (uint64_t)local.ids.size();

            direct_refine_clusters.clear();
            BucketSplitStats splitst = split_bins_singleton_vs_nonsingleton(
                subidx, bins_buckets, bins_buckets_nonsingleton, direct_refine_clusters, cfg.singleton_direct_refine);

            singleton_bucket_hits_total_t += splitst.singleton_bucket_hits;
            singleton_clusters_direct_refine_total_t += splitst.singleton_clusters_direct;
            nonsingleton_clusters_centroid_path_total_t += splitst.nonsingleton_clusters_centroid_path;

            const float* q_cent = logits.data() + (size_t)qi * (size_t)cfg.sub_centroids_d;

            auto tc0 = WallClock::now();
            heap.reset(cfg.centroid_heap_k);
            centroid_selected_clusters.clear();
            merged_refine_clusters.clear();

            CalibCsvStopDiag rr_calib_diag;
            CentroidStageResult rr = run_centroid_stage_copy_calibcsv_one_query(
                cfg, calib_tab, subidx, q_cent, bins_buckets_nonsingleton, heap,
                tmp_batch_f32.data(), tmp_dist_f32.data(), spans_tmp, &tmp_cids_buf, &rr_calib_diag);

            centroid_expand_ms_sum_t      += rr.expand_ms;
            centroid_gather_ms_sum_t      += rr.gather_ms;
            centroid_kernel_ms_sum_t      += rr.kernel_ms;
            centroid_sink_ms_sum_t        += rr.sink_ms;
            centroid_kernel_calls_t       += rr.kernel_calls;
            centroid_kernel_clusters_total_t += rr.clusters_seen;
            clusters_enum_total_t         += rr.clusters_seen;
            centroid_spans_before_total_t += rr.spans_before;
            centroid_spans_after_total_t  += rr.spans_after;

            heap_push_attempts_total_t += rr.heap_push_attempts;
            heap_updates_total_t += rr.heap_updates;
            heap_rejects_total_t += rr.heap_rejects;
            heap_appends_total_t += rr.heap_appends;
            heap_replaces_total_t += rr.heap_replaces;

            // stage output is now REAL sub-centroid IDs
            centroid_selected_clusters = heap.cid;

            merged_refine_clusters.reserve(direct_refine_clusters.size() + centroid_selected_clusters.size());
            merged_refine_clusters.insert(merged_refine_clusters.end(),
                                         direct_refine_clusters.begin(), direct_refine_clusters.end());
            merged_refine_clusters.insert(merged_refine_clusters.end(),
                                         centroid_selected_clusters.begin(), centroid_selected_clusters.end());

            direct_refine_clusters_merged_total_t += (uint64_t)direct_refine_clusters.size();
            centroid_selected_clusters_merged_total_t += (uint64_t)centroid_selected_clusters.size();
            merged_refine_candidates_total_t += (uint64_t)merged_refine_clusters.size();

            heap_selected_total_t += (uint64_t)heap.d.size();
            if(heap.full()) ++heap_full_q_t;

            calib_q_eval_t += 1;
            if(rr.early_stopped) calib_q_triggered_t += 1;
            if(rr.heap_full_at_end) calib_q_heap_full_end_t += 1;
            else calib_q_heap_not_full_end_t += 1;

            calib_total_batches_t += rr_calib_diag.total_batches;
            calib_low_qual_batches_t += rr_calib_diag.low_qual_batches;

            if(rr.early_stopped){
                early_stop_q_t += 1;
                for(int bb = rr.bins_processed; bb < cfg.NBINS; ++bb){
                    if(!bins_buckets_nonsingleton[(size_t)bb].empty()) early_stop_bins_skipped_t += 1;
                }
            }

            centroid_ms_sum_t += ms_since(tc0);
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            S.bins_build_ms_sum += bins_build_ms_sum_t;
            S.centroid_ms_sum += centroid_ms_sum_t;
            S.centroid_expand_ms_sum += centroid_expand_ms_sum_t;
            S.centroid_gather_ms_sum += centroid_gather_ms_sum_t;
            S.centroid_kernel_ms_sum += centroid_kernel_ms_sum_t;
            S.centroid_sink_ms_sum += centroid_sink_ms_sum_t;
            S.centroid_kernel_calls += centroid_kernel_calls_t;
            S.centroid_kernel_clusters_total += centroid_kernel_clusters_total_t;

            S.bins_nonempty_total += bins_nonempty_total_t;
            S.buckets_enum_total += buckets_enum_total_t;
            S.clusters_enum_total += clusters_enum_total_t;

            S.singleton_bucket_hits_total += singleton_bucket_hits_total_t;
            S.singleton_clusters_direct_refine_total += singleton_clusters_direct_refine_total_t;
            S.nonsingleton_clusters_centroid_path_total += nonsingleton_clusters_centroid_path_total_t;
            S.direct_refine_clusters_merged_total += direct_refine_clusters_merged_total_t;
            S.centroid_selected_clusters_merged_total += centroid_selected_clusters_merged_total_t;
            S.merged_refine_candidates_total += merged_refine_candidates_total_t;

            S.heap_selected_total += heap_selected_total_t;
            S.heap_push_attempts_total += heap_push_attempts_total_t;
            S.heap_updates_total += heap_updates_total_t;
            S.heap_rejects_total += heap_rejects_total_t;
            S.heap_appends_total += heap_appends_total_t;
            S.heap_replaces_total += heap_replaces_total_t;
            S.heap_full_q += heap_full_q_t;

            S.early_stop_q += early_stop_q_t;
            S.early_stop_bins_skipped += early_stop_bins_skipped_t;

            S.calib_q_eval += calib_q_eval_t;
            S.calib_q_triggered += calib_q_triggered_t;
            S.calib_q_heap_full_end += calib_q_heap_full_end_t;
            S.calib_q_heap_not_full_end += calib_q_heap_not_full_end_t;
            S.calib_total_batches += calib_total_batches_t;
            S.calib_low_qual_batches += calib_low_qual_batches_t;

            S.centroid_spans_before_total += centroid_spans_before_total_t;
            S.centroid_spans_after_total += centroid_spans_after_total_t;
        }
    }

    S.t_enum_stage_ms = ms_since(t0);
    S.t_total_ms = ms_since(t_total0);
    return S;
}

int main(int argc, char** argv){
    Config cfg;

    for(int i=1; i<argc; ++i){
        if(std::string(argv[i]) == "--json" && i+1 < argc){
            apply_json(cfg, slurp(argv[i+1]));
            break;
        }
    }
    override_cli(cfg, argc, argv);

    if(cfg.repeats <= 0) cfg.repeats = 1;
    if(cfg.centroid_heap_k <= 0) cfg.centroid_heap_k = 1024;
    if(cfg.centroid_kernel_batch_clusters <= 0) cfg.centroid_kernel_batch_clusters = 4096;
    if(cfg.centroid_batch_mode.empty()) cfg.centroid_batch_mode = "copy";
    if(cfg.centroid_batch_mode != "copy"){
        std::cerr << "[WARN] forcing centroid_batch_mode=copy\n";
        cfg.centroid_batch_mode = "copy";
    }
    if(cfg.centroid_gate_mode != "calib_csv"){
        std::cerr << "[WARN] forcing centroid_gate_mode=calib_csv (only mode supported)\n";
        cfg.centroid_gate_mode = "calib_csv";
    }

#ifdef _OPENMP
    std::cout << "[omp] _OPENMP=" << _OPENMP
              << " max_threads(at_start)=" << omp_get_max_threads()
              << "\n";
#endif

    std::cout << "[cfg]"
              << " Q=" << cfg.Q
              << " D_logits=" << cfg.D_logits
              << " D_in=" << cfg.D_in
              << " Lsel=" << cfg.Lsel
              << " Rmax=" << cfg.Rmax
              << " NBINS=" << cfg.NBINS
              << " cap_per_bin=" << cfg.cap_per_bin
              << " repeats=" << cfg.repeats
              << " centroid_heap_k=" << cfg.centroid_heap_k
              << " centroid_kernel_batch_clusters=" << cfg.centroid_kernel_batch_clusters
              << " singleton_direct_refine=" << (cfg.singleton_direct_refine?1:0)
              << " centroid_gate_mode=" << cfg.centroid_gate_mode
              << " centroid_early_stop=" << (cfg.centroid_early_stop?1:0)
              << " centroid_early_stop_after_bins=" << cfg.centroid_early_stop_after_bins
              << " calib_margin_csv=" << (cfg.calib_margin_csv.empty() ? "<none>" : cfg.calib_margin_csv)
              << " calib_margin_quantile=" << cfg.calib_margin_quantile
              << " heap_gate_by_calib_csv=" << (cfg.heap_gate_by_calib_csv?1:0)
              << " calib_lowqual_use_heap_updates=" << cfg.calib_lowqual_use_heap_updates
              << " calib_heap_front_alpha=" << cfg.calib_heap_front_alpha
              << " logits_path=" << (cfg.logits_path.empty() ? "<missing>" : cfg.logits_path)
              << "\n";

    std::vector<double> enum_ms_list, total_ms_list, qps_list;
    enum_ms_list.reserve((size_t)cfg.repeats);
    total_ms_list.reserve((size_t)cfg.repeats);
    qps_list.reserve((size_t)cfg.repeats);

    for(int rep=0; rep<cfg.repeats; ++rep){
        RoundStats s = run_round(cfg);
        print_round_summary(cfg, s, rep);

        enum_ms_list.push_back(s.t_enum_stage_ms);
        total_ms_list.push_back(s.t_total_ms);
        qps_list.push_back((s.t_enum_stage_ms > 0.0) ? (1000.0 * (double)cfg.Q / s.t_enum_stage_ms) : 0.0);
    }

    std::cout << "[ROUNDS]"
              << " n=" << enum_ms_list.size()
              << " enum_ms_p50=" << pct_from_vec(enum_ms_list, 50)
              << " enum_ms_p90=" << pct_from_vec(enum_ms_list, 90)
              << " enum_ms_p99=" << pct_from_vec(enum_ms_list, 99)
              << " total_ms_p50=" << pct_from_vec(total_ms_list, 50)
              << " enum_qps_p50=" << pct_from_vec(qps_list, 50)
              << "\n";

    return 0;
}