baseDirForScriptSelf=$(cd "$(dirname "$0")"; pwd)
docker container run --shm-size 220G -p 8000:3000 -it \
-v /nvdatasets/imagenet:/app/dataset_mirror \
-v ${baseDirForScriptSelf}/source_code:/app/source_code_mirror \
--runtime=nvidia --rm signum_docker sh run.sh