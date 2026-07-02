from SPHash_base import AutoHash, load_config, transform_json, update_margin_position_vector
from utils.tools import stage_to_shm
import sys,os,time
SRC_DIR = os.path.dirname(os.path.abspath(__file__))      # <repo>/src
PROJ_ROOT = os.path.dirname(SRC_DIR)                      # <repo>
sys.path.insert(0, PROJ_ROOT)
import numpy as np
from Bigann.indexing_model_bigann import Indexing_Model as Model

class AutoHashBigann(AutoHash):
    def __init__(self,config, metric):
        super().__init__(config, metric)  # Call base init
        print("Bigann indexing initialized")
    
    def get_Autoencoder_class(self):
        return Model

    
    
if __name__ == '__main__':
    metric = 'euclidean'
    vector_dim = 128
    json_name = 'results_20260226_122245'
    base_path = "./src/Bigann/index/"
    data_path = '/path/to/training_data_root/09012025_Bigann_training/training_data/'

    # Neural network input
    
    query_file = data_path + 'scaled_query.npy'
    output_base = f'{base_path}{json_name}/'
    nn_input_file = f"{data_path}base.crop100M.full.scaled.npy"
    database_file = "/path/to/baseline/big-ann-benchmarks/data/bigann/base.1B.u8bin.crop_nb_100000000"

 
    # calcuate median and update config
    source_json_base = './src/Bigann/01192026_results/'
    source_json_path = f'{source_json_base}{json_name}.json'
    target_json_name="Autohash_config.json"
    target_json_path = transform_json(source_json_path, target_json_name, base_path, vector_dim)
    config = load_config(target_json_path)

    at = AutoHashBigann(config, metric)
    
    out = at.compare_models_from_config(
        subset_x_npy_path=nn_input_file,
        topk=5,
        read_ratio=0.1,          
        work_dir="/dev/shm/sel",
        save_json_path=target_json_path,
        build_next_inputs=False,  
        cleanup_tmp=True,         
        target_autohash_config_path=target_json_path,
        write_back_model_path=True,
        round_ndigits=None,
    )
    
    config = load_config(target_json_path)

    at = AutoHashBigann(
    config,
    metric="l2",
)

    at.build_index(x_npy_path=nn_input_file,
    enc_out="/dev/shm/encoded_1b_22_f32.npy",
    next_out_dir="/dev/shm/csr_index",
    stage_json_path=target_json_path)
    
    prep = at.prepare_cpp_inputs_and_run(
    next_out_dir="/dev/shm/csr_index",
    vec_npy="/dev/shm/encoded_1b_22_f32.npy",  
    D=20,
    exe="./c_source_code/bucket_cluster/bucket_cluster_pipeline_v6_8_1_gt",
    threads=24,
    blas_threads=1,
)

    os.remove("/dev/shm/encoded_1b_22_f32.npy")
    with stage_to_shm([database_file], prefix="autohash_", verbose=True) as (staged, shm_dir):
        database_file_shm = staged[database_file]        
        out_subcsr = at.build_subcodes_from_next_dir(
        base_dir=output_base,                 
        vectors_path=database_file_shm,        
        d=vector_dim,                                 
        next_out_dir="/dev/shm/csr_index/subcsr",                                      
        vectors_format="u8_auto",
        out_codes_dtype=np.uint8,
        pos_block=2_000_000,
        io_buffer_mb=16,
        stage_json_path=target_json_path,
        stage_name="sub_codes_build",       
        verbose=True,
    )


    
