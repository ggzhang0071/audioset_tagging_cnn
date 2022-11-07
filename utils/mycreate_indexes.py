import os
import h5py

def create_indexes(args):
    """Create indexes a for dataloader to read for training. When users have 
    a new task and their own data, they need to create similar indexes. The 
    indexes contain meta information of "where to find the data for training".
    """
    waveforms_hdf5_path=args.waveforms_hdf5_path
    indexes_hdf5_path=args.indexes_hdf5_path

    index_dir= os.path.dirname(indexes_hdf5_path)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    elif os.path.exists(indexes_hdf5_path):
        os.remove(indexes_hdf5_path)
    
    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S100')
            hw.create_dataset('target', data=hr['target'][:], dtype=bool)
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            # why create the index_in_hdf5?
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)
    print('Write to {}'.format(indexes_hdf5_path))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    subparsers=parser.add_subparsers(dest="mode")
    parser_create_indexes=subparsers.add_parser("create_indexes")
    parser_create_indexes.add_argument('--waveforms_hdf5_path', type=str, required=True)
    parser_create_indexes.add_argument('--indexes_hdf5_path', type=str, required=True)
    args = parser.parse_args()
    if args.mode=="create_indexes":
        create_indexes(args)
    else:
        raise Exception("Error mode!")


