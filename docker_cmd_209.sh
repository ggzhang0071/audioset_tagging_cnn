#docker_cmd_93.sh
img="nvcr.io/nvidia/pytorch:19.03-py3"
#nvcr.io/nvidia/pytorch:18.01-py3



docker run --gpus all  --privileged=true   --workdir /git --name "atcnn"  -e DISPLAY --ipc=host -d --rm  -p 6610:4452  \
-v /home/zgg/audioset_tagging_cnn:/git/audioset_tagging_cnn \
 -v  /home/zgg/datasets:/git/datasets \
 $img sleep infinity


docker exec -it atcnn   /bin/bash

cd audioset_tagging_cnn/  

#pip install -r COSMIC/requirements.txt
#docker images |grep pytorch  | grep  "19"