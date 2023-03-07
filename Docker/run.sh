#!/bin/bash
if [ "$#" -le 1 ]; then
  echo "Need at least two arguments: container name and path to mount. And optionally 'cpu' for using cpu mode."
elif [ "$#" = 2 ] || [ "$3" != "cpu" ]; then
    docker create -it -e "new_uid"=$(id --user) -e "new_gid"=$(id --group) -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" -e "XAUTHORITY=${XAUTH}" -v ~/.Xauthority:/home/docker/.Xauthority:rw -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /dev:/dev -v /etc/hosts:/etc/hosts --network host --privileged --runtime=nvidia --name $1 -v $2:/home/docker/actvh_ws rustlluk/vishac >> /dev/null
    docker start $1 >> /dev/null && docker exec --user root $1 change_uid.sh >> /dev/null && docker stop $1 >> /dev/null && docker start $1 >> /dev/null && docker attach $1
else
    docker create -it -e "new_uid"=$(id --user) -e "new_gid"=$(id --group) -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" -e "XAUTHORITY=${XAUTH}" -v ~/.Xauthority:/home/docker/.Xauthority:rw -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /dev:/dev -v /etc/hosts:/etc/hosts --network host --privileged --name $1 -v $2:/home/docker/actvh_ws rustlluk/vishac >> /dev/null
    docker start $1 >> /dev/null && docker exec --user root $1 change_uid.sh >> /dev/null && docker stop $1 >> /dev/null && docker start $1 >> /dev/null && docker attach $1
fi
