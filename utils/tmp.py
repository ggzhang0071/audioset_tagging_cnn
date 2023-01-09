import pandas as pd
import os,json
import librosa
def collect_data_from_ef(test_wav_dir,original_json_path,save_json_path,max_label_num):
    data_info=pd.read_csv(original_json_path,sep="\t")
    label_map = {"male": 0, "female": 1, "child": 2,"baby": 3}
    choosed_wav_info=[]
    label_counter={"male": 0, "female": 0, "baby": 0, "child": 0}
    for idx, row in data_info.iterrows():
        if ("male" in row[1]) or ("female"  in row[1]) or ("baby" in row[1]) or ("child" in row[1]):
            # check the wav file 
            wav_file=os.path.join(test_wav_dir,row[0])
            try:
                waveform, sample_rate=librosa.load(wav_file, sr=16000)
            except:
                #print("{} error".format(wav_file))
                continue
            else:
                time = librosa.get_duration(filename=wav_file)
                if(time>0):
                    type1=row[1].split("_")[0]
                    assert type1 in ["male", "female", "baby", "child"]
                    if label_counter[type1]<max_label_num:
                        print("label_counter", label_counter)
                        choosed_wav_info.append({"wav":wav_file,"labels":label_map[type1],"display_name":type1.capitalize()})
                        label_counter[type1]+=1
    # save the choosed files
    with open(save_test_json, 'w') as f:
        json.dump({"data":choosed_wav_info}, f,indent=4)
    print("Finish the test ef audio dataset preparation")

if __name__ == "__main__":
        test_wav_dir="/git/datasets/audio_ef_wav/shujutang_wav12"
        original_json_path="/git/datasets/audio_ef_wav/shujutang_wav12_list_txt"
        save_test_json="/git/datasets/audio_ef_wav/chooosed_test_human_sounds.json"
        max_label_num=4
        collect_data_from_ef(test_wav_dir,original_json_path,save_test_json,max_label_num)  
