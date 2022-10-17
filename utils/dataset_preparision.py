## data preparation for ef envorment soud classification
from glob import  iglob
import json, os, random
import pandas as pd
import librosa 
import numpy as np
import   config
import logging
import math
import copy 
import argparse


def make_new_class_label_indices(choosed_types,save_json):
    # save new class label indices to csv file
    df=pd.DataFrame(columns=["index","mid","display_name"])
    if isinstance(choosed_types,pd.DataFrame):
        for i,row in choosed_types.iterrows():
            df.loc[i]=[i, row["index"], row["type"]]
    elif isinstance(choosed_types,dict):
        for i, (type, mid) in enumerate(choosed_types.items()):
            df.loc[i]=[i,mid,type]
    else:
        raise Exception("The choosed_types isn't surpport, it should be a dict or pd.DataFrame")
    df.to_csv(save_json,index=False)

def choose_intersection_label(str1,choosed_types):
    intersection_label=""
    for label in choosed_types["index"]:
        if label in str1:
            intersection_label+=label+" "
    return intersection_label

# compute the mean and std of child ef dataset
def compute_mean_std(wav_dir):
    mean_list=[]
    std_list=[]
    counter=0
    for wav_file in iglob(wav_dir+"/*.wav"):
        wav_data, _ = librosa.load(wav_file, sr=None)
        mean_list.append(np.mean(wav_data))
        std_list.append(np.std(wav_data))
        counter+=1
        if counter%2000==0:
            print("The counter is {}".format(counter))

    print("The ef_audio dataset mean  is {}, std is {}".format(np.mean(mean_list),np.mean(std_list)))
    #The  ef_audio dataset mean  is -2.0259456505300477e-05, std is 0.019023671746253967



def collect_data_from_audioset(csv_dir,choosed_types,save_test_json,multi_label):
    #Find the wav file according the wav file, since there are some csv file are not consistent with the wav file
    wav_label_list=[]
    parent_label=0
    for parent_name, types in choosed_types.items():    
        counter=0
        for index_name_pair in types:
            # check the csv file and wav file
            file1=list(index_name_pair.values())[0]
            index=list(index_name_pair.keys())[0]
            csv_path=os.path.join(csv_dir,file1+".csv")
            wav_path=os.path.join(csv_dir,file1)
            if not os.path.exists(csv_path):
                raise Exception("The csv file {} isn't exist, please check it ".format(csv_path))
            if not os.path.exists(wav_path):
                raise Exception("The wav file {} isn't exist, please check it".format(wav_path))

            # list all the wav file
            for csv_file in iglob(csv_path):
                df=pd.read_csv(csv_file)
                # the format of the wav file writed according the csv file format 
                wav_names=df.iloc[:,0]+"_"+df.iloc[:,1].astype(int).astype(str) 
        
            # list all wav file and its label into json file
            for wav_file in iglob(wav_path+"/*.wav"):
                try:
                    waveform, sample_rate=librosa.load(wav_file, sr=16000)
                except:
                    #print("{} error".format(wav_file))
                    continue
                else:
                    time = librosa.get_duration(filename=wav_file)
                    if(time>0):
                        wav_name=os.path.splitext(wav_file.split("/")[-1])[0]
                        for i, wav_name_ in enumerate(wav_names):
                            #if wav_name==wav_name_ and index in df.iloc[i,3]:
                            if wav_name==wav_name_:
                                # multi label for multi-task ??
                                if multi_label:
                                    label=choose_intersection_label(df.iloc[i,3],choosed_types)
                                else:
                                    label=index
                                wav_label_list.append({"wav":wav_file,"labels":parent_label,"display_name":parent_name})
                                break
                #print("The type {} num is {}".format(file1,len(wav_label_list)))
        parent_label+=1
    print("The total type num is {}".format(len(tuple(wav_label_list))))

    with open(save_test_json, 'w') as f:
        json.dump({"data":wav_label_list}, f,indent=4)
        print("Finish the part audioset dataset preparation")

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


def merge_json(json_folder,first_json,second_json):
    # merge the json file
    with open(os.path.join(json_folder,first_json), 'r') as f:
        first_json_data=json.load(f)
    with open(os.path.join(json_folder,second_json), 'r') as f:
        second_json_data=json.load(f)       
    first_json_data["data"].extend(second_json_data["data"])
    with open(os.path.join(json_folder,first_json), 'w') as f:
        json.dump(first_json_data, f,indent=4)
        print("Finish the merge json file")
    # delete the second json file
    os.remove(os.path.join(json_folder,second_json))

def collect_data_from_esc50(dataset_json_folder,choosed_types,save_dir,save_val_json):
    new_wav_label_list=[]
    for dataset_json_file in iglob(dataset_json_folder+"/*.json"):
        print("The current json file is {}".format(dataset_json_file))
        #datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        data = data_json['data']
        for wav_label in data:
            if wav_label["labels"] in choosed_types.values():
                new_wav_label_list.append(wav_label)
    # save the new wav-label pair     

    with open(os.path.join(save_dir,save_val_json), 'w') as f:
        json.dump({"data":new_wav_label_list}, f, indent=4)
        print("Finish the part esc50 dataset preparation")

def split_data(save_dir,save_test_json,k_fold, split_ratio,dataset,mode="train"):
    with open(os.path.join(save_dir,save_test_json), 'r') as f:
        data_json = json.load(f)
        data=data_json["data"]
        len_dataset=(len(data))

    data_list=list(range(len_dataset))
    random.shuffle(data_list)
    length=int(len_dataset/k_fold)

    chunks = [data_list[x:x+length] for x in range(0, len_dataset, length)]

    for i, fold in enumerate(range(1,k_fold+1)):
        train_wav_list=[]
        val_wav_list=[]
        for j,chunk in enumerate(chunks[i]):
            if j<=int(length*split_ratio):
                train_wav_list.append(data[chunk])
            else:
                val_wav_list.append(data[chunk])
        if mode=="train":
            print("The {} dataset train wav num is {}, and the validate wav num is {}".format(dataset,len(train_wav_list),len(val_wav_list)))
            with open(save_dir+'/part_{}_train_data_{}.json'.format(dataset,fold), 'w') as f:
                json.dump({'data': train_wav_list}, f, indent=4)

            with open(save_dir+'/part_{}_eval_data_{}.json'.format(dataset,fold), 'w') as f:
                json.dump({'data': val_wav_list}, f, indent=4)

        elif mode=="test":
            print("The {} dataset validate wav num is {}, and the test wav num is {}".format(dataset,len(train_wav_list),len(val_wav_list)))

            with open(save_dir+'/part_{}_eval_data_{}.json'.format(dataset,fold), 'w') as f:
                json.dump({'data': train_wav_list}, f, indent=4)

            with open(save_dir+'/part_{}_test_data_{}.json'.format(dataset,fold), 'w') as f:
                json.dump({'data': val_wav_list}, f, indent=4)
    print("Finish the part {} preparation".format(dataset))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='audioset', help='train and dataset name')
    parser.add_argument('--test_dataset', type=str, default='ef_audio', help='test dataset name')
    parser.add_argument('--merged',action='store_false',help='merged types or not')
    parser.add_argument('--class_labels_indices',type=str,default="/git/audioset_tagging_cnn/metadata/class_labels_indices.csv")
    parser.add_argument("--k_fold", type=int, default=1, help="the k fold for cross validation")
    #parser.add_argument("--choosed_types", type=dict, default={}, help="the choosed types")
    parser.add_argument("--multi_label", action="store_true",  help="using the  multi labels")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="the split ratio")
    parser.add_argument("--save_dir", type=str, default="/git/datasets/from_audioset/datafiles", help="json file save dir")
    parser.add_argument("--csv_dir", type=str, default="/git/datasets/from_audioset", help="csv file save dir")
    parser.add_argument("--label_csv", type=str, default="/git/datasets/from_audioset/datafiles/part_audioset_class_labels_indices.csv", help="json file for tarin and val")
    parser.add_argument("--test_label_csv", type=str, default="/git/datasets/from_audioset/datafiles/part_esc50_class_labels_indices.csv",required=False, help="json file for test")
    parser.add_argument("--test_split", action="store_true", help="split test dataset")

    args=parser.parse_args()
    save_dir=args.save_dir
    test_dataset=args.test_dataset
    os.makedirs(save_dir, exist_ok=True)
    os.system("rm -rf "+save_dir+"/*")

    if args.dataset=="audioset":
        # collect the choosed type from audiosets
        original_choosed_types={'Baby cry, infant cry':"Babycryinfantcry", 'Baby laughter': "Babylaughter", 'Child singing': "Childsinging",'Children shouting': "Childrenshouting", 'Children playing':"Childrenplaying", 'Child speech, kid speaking': "Childspeechkidspeaking",
       'Female singing':  "Femalesinging", 'Female speech, woman speaking':"Femalespeechwomanspeaking", 'Male singing': "Malesinging", 'Male speech, man speaking': "Malespeechmanspeaking"}
        saved_choosed_types=copy.deepcopy(original_choosed_types)
        df=pd.read_csv(args.class_labels_indices)
        if args.merged==True:
            classification_types=config.classification_types
            new_classification_types={}
            for type1 in  reversed(classification_types):
                for sub_type in reversed(list(original_choosed_types.keys())):
                    if type1 in sub_type:
                        loc=df[df["display_name"]==sub_type].index.tolist()
                        index=df.iloc[loc,1].values[0]
                        index_name_pair={index:original_choosed_types[sub_type]}
                        if type1 not in  new_classification_types.keys():
                            new_classification_types[type1]=[index_name_pair]
                        else:
                            new_classification_types[type1].append(index_name_pair)
                        original_choosed_types.pop(sub_type,None)
        
        save_test_json= os.path.join(save_dir,"part_{}_data.json".format(args.dataset))
        #make_new_class_label_indices(new_classification_types,args.label_csv)
        collect_data_from_audioset(args.csv_dir,new_classification_types,save_test_json,args.multi_label)
        split_data(save_dir,save_test_json,args.k_fold,args.split_ratio,args.dataset)
    elif args.dataset=="esc50":
        # collect data from esc50
        #choosed_types={"crying_baby":"/m/07rwj20","laughing":"/m/07rwj26","mouse_click":"/m/07rwj31","keyboard_typing":"/m/07rwj32"}
        choosed_types={"crying_baby":"/m/07rwj20","laughing":"/m/07rwj26","keyboard_typing":"/m/07rwj32","mouse_click":"/m/07rwj31"}
        save_test_json="choose_data_from_esc50.json"
        original_json_dir='/git/datasets/esc50/datafiles'
        make_new_class_label_indices(choosed_types,args.label_csv)
        collect_data_from_esc50(original_json_dir,choosed_types,save_dir,save_test_json)
        split_data(save_dir,save_test_json,args.k_fold,args.split_ratio,args.dataset)
    else:
        #raise ValueError("we now only support the audioset and esc50 dataset")
        pass

    if test_dataset=="esc50":
        # collect the data from esc50
        #choosed_types={"crying_baby":"/m/07rwj20","laughing":"/m/07rwj26","mouse_click":"/m/07rwj31","keyboard_typing":"/m/07rwj32"}
        test_choosed_types={"crying_baby":"/m/07rwj20","keyboard_typing":"/m/07rwj32","laughing":"/m/07rwj26","mouse_click":"/m/07rwj31"}
        if dataset=="esc50":
            new_types_keys=choosed_types.keys() ^ test_choosed_types.keys()
            split_ratio=0.5
            for fold in range(1,k_fold+1):
                save_eval_json="part_esc50_eval_data_{}.json".format(fold)
                save_test_json="part_esc50_test_data_{}.json".format(fold)
                split_data(save_dir,save_eval_json,1,split_ratio,test_dataset,mode="test")
            if len(new_types_keys)>0:
                new_types={key:test_choosed_types[key] for key in new_types_keys}
                # if the dataset is esc50, split the validation set to validate and test
                original_json_dir='/git/datasets/esc50/datafiles'
                #make_new_class_label_indices(choosed_types,test_label_csv)
                save_part_test_json="new_types_from_esc50.json"
                collect_data_from_esc50(original_json_dir,new_types,save_dir,save_part_test_json)
                merge_json(save_dir,save_test_json,save_part_test_json)
        else:
            choosed_types=test_choosed_types
            save_test_json="choose_data_from_esc50.json"
            original_json_dir='/git/datasets/esc50/datafiles'
            make_new_class_label_indices(choosed_types,args.test_label_csv)
            collect_data_from_esc50(original_json_dir,choosed_types,save_dir,save_test_json)
            split_data(save_dir,save_test_json,args.k_fold,args.split_ratio,test_dataset)

    elif test_dataset=="ef_audio":
        test_wav_dir="/git/datasets/audio_ef_wav/shujutang_wav12"
        original_json_path="/git/datasets/audio_ef_wav/shujutang_wav12_list_txt"
        save_test_json="/git/datasets/audio_ef_wav/chooosed_human_sounds.json"
        max_label_num=4
        #compute_mean_std(args.test_wav_dir)
        if args.test_split:
            split_data(save_dir,save_test_json,k_fold,split_ratio,test_dataset)
    else:
        raise Exception("The dataset isn't supported")

    
     
        

  

   


