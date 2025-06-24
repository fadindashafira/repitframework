# Base: PyTorch with CUDA (compatible with CUDA driver 12.4)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Environment and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# System dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    vim \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 1. Add user `ninelab`
RUN useradd -ms /bin/bash ninelab && \
    usermod -aG sudo ninelab && \
    echo "ninelab ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 2. Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -u -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Default env activation
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=base
ENV PATH /opt/conda/envs/base/bin:$PATH

# 3. Install OpenFOAM v12
RUN wget -q -O - https://dl.openfoam.org/gpg.key | apt-key add - && \
    add-apt-repository http://dl.openfoam.org/ubuntu && \
    apt-get update && \
    apt-get install -y openfoam12 && \
    echo "source /opt/openfoam12/etc/bashrc" >> /home/ninelab/.bashrc && \
    chown -R ninelab:ninelab /opt/openfoam12 /home/ninelab

# 4. Environment variables
ENV PATH="/opt/openfoam12/bin:$PATH"
ENV FOAM_RUN=/home/ninelab/run
ENV FOAM_INST=/opt/openfoam12
ENV DISPLAY=:0

# 5. Working dir and mountable volume
WORKDIR /home/ninelab
VOLUME /home/ninelab/repitframework

# Run as non-root
USER ninelab

CMD ["/bin/bash"]
