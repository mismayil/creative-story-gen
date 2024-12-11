FROM --platform=linux/amd64 nvidia/cuda:12.2.2-runtime-ubuntu22.04


# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ENV USER_NAME=ismayilz
ENV HOME=/home/${USER_NAME}
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV REPO_DIR=${HOME}/project-csg
ENV CONDA_ENV=cs-gen

WORKDIR /

# Install dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN apt update && apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        git \
        curl \
        vim \
        unzip \
        wget \
        tmux \
        screen \
        ca-certificates \
        apt-utils \
        libjpeg-dev \
        libpng-dev \
        sudo \
        htop \
        apt-transport-https \
        gnupg \
        protobuf-compiler \
        gfortran \
        libopenblas-dev \
        liblapack-dev

# Set up user
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    groupadd -g ${NLP_GROUP_ID} ${NLP_GROUP_NAME} && \
    useradd -r -m -d /home/${USER_NAME} -s /bin/bash -g ${GROUP_ID} -G ${NLP_GROUP_NAME},sudo -u ${USER_ID} ${USER_NAME} && \
    # Change the password to make root > user
    echo -e "${USER_NAME}\n${USER_NAME}" | passwd ${USER_NAME}

# Set up sudo users
RUN echo 'root:root' | chpasswd
RUN echo -e "\n${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

# Prepare the NFS mount folders
RUN mkdir /mnt/nlpdata1
RUN mkdir /mnt/scratch

# Git configuration (optional, you can also use a repo saved in the NFS)
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    git config --global user.name ${GITHUB_NAME}
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    git config --global user.email ${GITHUB_EMAIL}
RUN git config --global pull.rebase false

# Switch to home directory
WORKDIR ${HOME}
USER ${USER_NAME}

# Install conda (optional)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN ${CONDA} create -y --name ${CONDA_ENV} python=3.10

# Setup github repo dependencies
COPY --chown=${USER_NAME} ./requirements.txt .
RUN ${CONDA} run -n ${CONDA_ENV} pip install -r ./requirements.txt
RUN ${CONDA} run -n ${CONDA_ENV} pip install ipykernel ipywidgets

COPY --chown=${USER_NAME} ./docker/entrypoint.sh .
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/bin/bash"]

# ENTRYPOINT ["sudo", "/usr/sbin/sshd", "-D"]
