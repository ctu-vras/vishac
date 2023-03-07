#!/bin/bash
old_uid=$( id -u docker)
old_gid=$( id -g docker)
sudo sed -i 's|docker:x:'"$old_uid"':'"$old_gid"'::/home/docker:/bin/bash|docker:x:'"$new_uid"':'"$new_gid"'::/home/docker:/bin/bash|g' /etc/passwd
sudo find /home/docker -group $old_gid -not -path "/home/docker/actvh_ws*" -exec chgrp -h $new_gid {} \;
sudo find /home/docker -user $old_uid -not -path "/home/docker/actvh_ws*" -exec chown -h $new_uid {} \;
