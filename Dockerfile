FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Create a new user openfoam
RUN useradd -ms /bin/bash openfoam && \
    usermod -aG sudo openfoam && \
    echo "openfoam ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set environment variables for non-interactive installations and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install OpenFOAM
RUN apt-get update && apt-get install -y software-properties-common wget sudo vim && \
    wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc &&\
    add-apt-repository http://dl.openfoam.org/ubuntu &&\
    apt-get update &&\
    apt-get install -y openfoam12 &&\
    echo "source /opt/openfoam12/etc/bashrc" >> /home/openfoam/.bashrc &&\
    chown -R openfoam:openfoam /opt/openfoam12 /home/openfoam

# Install Python packages
RUN conda install -y -c conda-forge numpy matplotlib pandas &&\
    pip install Ofpp


# Set environment variables
ENV PATH=/opt/openfoam12/bin:$PATH
ENV FOAM_RUN=/home/openfoam/run
ENV FOAM_INST=/opt/openfoam12

# Set the DISPLAY environment variable for GUI support
ENV DISPLAY=:0

# Specify the volume for host mounting:
VOLUME /home/openfoam/host_mount

CMD ["/bin/bash"]