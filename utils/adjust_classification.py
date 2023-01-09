import json, os 

# merge the baby class into the child class
json_dir="/git/datasets/from_audioset/datafiles_ok"

eval_json="part_audioset_eval_data_1.json"
train_json="part_audioset_train_data_1.json"
test_json="chooosed_test_human_sounds.json"

save_new_train_json="part_audioset_train_data_1_3.json"
save_new_eval_json="part_audioset_eval_data_1_3.json"
save_new_test_json="chooosed_test_human_sounds_3.json"



json_file_list ={train_json:save_new_train_json, eval_json:save_new_eval_json, test_json:save_new_test_json}

for json_file in  json_file_list.keys():
    json_path=os.path.join(json_dir,json_file)
    with open(json_path,"r") as f:
        json_data=json.load(f)
        json_list = json_data["data"]
        for i  in range(len(json_list)):
            if json_list[i]["display_name"] =="Baby":
                json_list[i]["display_name"]="Child"
                json_list[i]["labels"]=2
    with open(os.path.join(json_dir,json_file_list[json_file]),"w") as fid:
        json.dump({"data":json_list}, fid,indent=4)




