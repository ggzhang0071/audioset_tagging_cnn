#docker_cmd_93.sh
img="nvcr.io/nvidia/pytorch:21.08-py3"
#nvcr.io/nvidia/pytorch:18.01-py3


docker run --gpus all  --privileged=true   --workdir /git --name "atcnn_21"  -e DISPLAY --ipc=host -d --rm  -p 6621:4452  \
-v /home/zgg/audioset_tagging_cnn:/git/audioset_tagging_cnn \
-v  /srv/ai-team/datasets:/git/datasets \
$img sleep infinity

docker exec -it atcnn_21 /bin/bash 


#pip install -r COSMIC/requirements.txt
#docker images |grep pytorch  | grep  "21"
