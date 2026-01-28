# 包含 DTK 的镜像
IMAGE_NAME="harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.4-1230-py3.10-20251230"
#IMAGE_NAME="image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.2-py3.10"
CONTAINER_NAME=one-matpl

docker pull $IMAGE_NAME

docker run -it \
--name=$CONTAINER_NAME \
-v /opt/hyhal:/opt/hyhal:ro \
-v $PWD:/workspace \
-w /workspace \
--hostname=localhost \
--network=host \
--ipc=host \
--device=/dev/kfd \
--device=/dev/mkfd \
--device=/dev/dri \
--shm-size=512G \
--privileged \
--group-add video \
--cap-add=SYS_PTRACE \
-u root \
--security-opt seccomp=unconfined \
$IMAGE_NAME \
/bin/bash
