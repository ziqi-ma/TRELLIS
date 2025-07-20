import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--original_3d', type=str, default='original_3d',
                        help='Folder containing the original 3D model')
    parser.add_argument('--edited_3d', type=str, default='edited_3d',
                        help='Folder containing the edited 3D model')
    parser.add_argument('--metadata_path', type=str, default='metadata.csv',
                        help='CSV file containing the shard metadata')


def get_metadata(original_3d, edited_3d, metadata_path, **kwargs):
    seen = set()
    columns = ['sha256', 'file_identifier', 'aesthetic_score', 'captions']
    df = pd.DataFrame(columns=columns)
    
    metadata = pd.read_csv(metadata_path)
    for idx, row in metadata.iterrows():
        if row["source"] not in seen:
            if row["type"] == "forward":
                df.loc[len(df)] = [
                    row["source"],
                    os.path.join(original_3d, row["source"], "mesh_textured.glb"),
                    None, None
                ]
            elif row["type"] == "backward":
                df.loc[len(df)] = [
                    row["source"],
                    os.path.join(edited_3d, row["source"], "mesh_textured.glb"),
                    None, None
                ]
            
            seen.add(row["source"])
        elif row["target"] not in seen:
            if row["type"] == "forward":
                df.loc[len(df)] = [
                    row["target"],
                    os.path.join(edited_3d, row["target"], "mesh_textured.glb"),
                    None, None
                ]
            elif row["type"] == "backward":
                df.loc[len(df)] = [
                    row["target"],
                    os.path.join(original_3d, row["target"], "mesh_textured.glb"),
                    None, None
                ]
            seen.add(row["target"])
    
    return df
        

def download(metadata, output_dir, **kwargs):
    metadata['local_path'] = metadata['file_identifier']

    return metadata[['sha256', 'local_path']]


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    file = metadatum["local_path"]
                    record = func(file, metadatum["sha256"])
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {metadatum["sha256"]}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
