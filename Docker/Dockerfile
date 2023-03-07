FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
# start with the nvidia container for latest CUDA and cudnn

LABEL maintainer="Lukas Rustler <rustlluk@fel.cvut.cz>"

#Non-interactive mode
ENV DEBIAN_FRONTEND='noninteractive'

#First update
RUN rm -r /etc/apt/sources.list.d/*cuda*
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install wget sudo -y 

#Update and install all
RUN apt-get update
RUN apt-get install unzip apt-utils curl python3.6 python3-distutils python3-venv python3-opengl gedit nano libsm6 \
     libxext6 libxrender-dev git cmake dos2unix firefox xfce4 xfce4-goodies \
     xserver-xorg-video-dummy xserver-xorg-legacy x11vnc gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-base software-properties-common locales libgl1-mesa-glx libgl1-mesa-dri mesa-utils \
    libeigen3-dev libcppunit-dev doxygen graphviz python3-psutil python3-future python3-sphinx -y

# Python 3.6
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip3 install --ignore-installed opencv-python plyfile scikit-image trimesh torch open3d scipy pyhocon plotly pymeshfix conan


#ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN apt update && apt install ros-noetic-desktop-full python3-rosdep python3-rosinstall python3-rosinstall-generator \
     python3-wstool build-essential python3-rosdep python3-catkin-tools ros-noetic-moveit* \
     ros-noetic-ddynamic-reconfigure* librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg \
     libglfw3-dev htop net-tools sshfs screen ros-noetic-effort* ros-noetic-ros-control* ros-noetic-rgbd-launch -y

#Set locale
RUN locale-gen en_US.UTF-8

#Graphics
RUN apt remove -y xfce4-power-manager light-locker && \
    sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config

#XORG setup
COPY xorg.conf /etc/X11/xorg.conf
RUN dos2unix /etc/X11/xorg.conf

#noVNC
WORKDIR /root/git
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify && \
    echo "<html><head><meta http-equiv=\"Refresh\" content=\"0; url=vnc.html?autoconnect=true&reconnect=true&reconnect_delay=1000&resize=scale&quality=9\"></head></html>" > /opt/novnc/index.html

COPY start-vnc-session.sh /usr/bin/start-vnc-session.sh
RUN chmod +x /usr/bin/start-vnc-session.sh && \
    dos2unix /usr/bin/start-vnc-session.sh

EXPOSE 5901 6080 10000/tcp 10000/udp
#Browser settings
ENV DONT_PROMPT_WSL_INSTALL=1
ENV VNC_DISPLAY=:1
ENV DISPLAY=:1
RUN update-alternatives --install /usr/bin/x-www-browser x-www-browser /usr/bin/firefox 500
RUN update-alternatives --set x-www-browser /usr/bin/firefox

# Create user gitpod
RUN useradd -l -u 1000 -G sudo -md /home/docker -s /bin/bash -p docker docker && \
    # passwordless sudo for users in the 'sudo' group
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

COPY change_uid.sh /usr/bin/change_uid.sh
RUN chmod +x /usr/bin/change_uid.sh && \
    dos2unix /usr/bin/change_uid.sh

# YOLOv7
RUN pip3 install -r "https://raw.githubusercontent.com/WongKinYiu/yolov7/main/requirements.txt" --disable-pip-version-check

USER docker
#Prepare git dump folder
WORKDIR /home/docker

# Aditional things
ADD binvox_rw.py /usr/lib/python3.8/
COPY mujoco210 /home/docker/.mujoco/mujoco210
RUN sudo ln -s /home/docker/actvh_ws/src/v-hacd/build/linux2/test/testVHACD /usr/bin/testVHACD
RUN sudo ln -s /home/docker/actvh_ws/src/Fast-Robust-ICP/build/FRICP /usr/bin/FRICP
RUN sudo ln -s /home/docker/actvh_ws/src/shape_completion/scripts/run_simulation.sh /usr/bin/run_simulation
RUN wget http://www.patrickmin.com/binvox/linux64/binvox && chmod a+x binvox && mkdir -p /home/docker/.local/bin && \
     mv binvox /home/docker/.local/bin && sudo ln -s /home/docker/.local/bin/binvox /usr/bin/binvox

# Kortex
RUN echo "source /home/docker/actvh_ws/devel/setup.bash" >> /home/docker/.bashrc

RUN echo "alias vnc='export DISPLAY=\$VNC_DISPLAY'" >> /home/docker/.bashrc

#Delete trash
CMD ["bash"]
