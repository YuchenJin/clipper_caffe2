FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get -y update && \
    apt-get -y install \
    build-essential \
      autotools-dev \
      rsync \
      curl \
      wget \
      jq \
      openssh-server \
      openssh-client \
      sudo \
      cmake \
      g++ \
      python-pip \
      gcc \
    # ifconfig
      net-tools  \
      iputils-ping  \
      vim \
      fish  \
      tmux && \
    apt-get autoremove

RUN apt-get install -y g++ automake wget autoconf autoconf-archive libtool libboost-all-dev \
    libevent-dev libdouble-conversion-dev libgoogle-glog-dev libgflags-dev liblz4-dev \
    liblzma-dev libsnappy-dev make zlib1g-dev binutils-dev libjemalloc-dev libssl-dev \
    pkg-config libiberty-dev git cmake libev-dev libhiredis-dev libzmq5 libzmq5-dev build-essential

## Install Folly
RUN git clone https://github.com/facebook/folly \
    && cd folly/folly \
    && git checkout tags/v2017.08.14.00 \
    && autoreconf -ivf \
    && ./configure \
    && make -j4 \
    && make install

## Install Cityhash
RUN git clone https://github.com/google/cityhash \
    && cd cityhash \
    && ./configure \
    && make all check CXXFLAGS="-g -O3" \
    && make install

COPY ./ /clipper

RUN  cd /clipper \
    && ./configure --cleanup-quiet \
    && ./configure --release \
    && cd release \
    && make -j8 management_frontend \
    && make -j8 query_frontend

# Install Redis.
RUN \
  cd /tmp && \
  wget http://download.redis.io/redis-stable.tar.gz && \
  tar xvzf redis-stable.tar.gz && \
  cd redis-stable && \
  make && \
  make install && \
  cp -f src/redis-sentinel /usr/local/bin && \
  mkdir -p /etc/redis && \
  cp -f *.conf /etc/redis && \
  rm -rf /tmp/redis-stable* && \
  sed -i 's/^\(bind .*\)$/# \1/' /etc/redis/redis.conf && \
  sed -i 's/^\(daemonize .*\)$/# \1/' /etc/redis/redis.conf && \
  sed -i 's/^\(dir .*\)$/# \1\ndir \/data/' /etc/redis/redis.conf && \
  sed -i 's/^\(logfile .*\)$/# \1/' /etc/redis/redis.conf

# Define mountable directories.
VOLUME ["/data"]

# Define working directory.
WORKDIR /data

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Define default command.
CMD ["redis-server", "/etc/redis/redis.conf"]

# Expose ports.
EXPOSE 6379
EXPOSE 1338
EXPOSE 1337
EXPOSE 7000

#RUN pip install numpy scikit-learn requests
RUN cd ~ && git clone https://github.com/YuchenJin/clipper_caffe2.git
                

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-9.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION} \
    OFED_VERSION=4.0-2.0.0.1

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get -y install \
    build-essential \
      autotools-dev \
      rsync \
      curl \
      wget \
      jq \
      openssh-server \
      openssh-client \
    # No longer in 'minimal set of packages'
      sudo \
    # Needed by OpenMPI
      cmake \
      g++ \
      gcc \
    # ifconfig
      net-tools  \
      iputils-ping  \
      vim \
      fish  \
      tmux && \
    apt-get autoremove

WORKDIR $STAGE_DIR

# Install Mellanox OFED user-mode drivers and its prereqs
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    # For MLNX OFED
        dnsutils \
        pciutils \
        ethtool \
        lsof \
        python-libxml2 \
        quilt \
        libltdl-dev \
        dpatch \
        autotools-dev \
        graphviz \
        autoconf \
        chrpath \
        swig \
        automake \
        tk8.4 \
        tcl8.4 \
        libgfortran3 \
        tcl \
        libnl-3-200 \
        libnl-route-3-200 \
        libnl-route-3-dev \
        libnl-utils \
        gfortran \
        tk \
        bison \
        flex \
        libnuma1 && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*

##################### OPENMPI #####################

RUN wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

#ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

#RUN ln -s /usr/include/cudnn.h /usr/local/cuda/include/
#RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/
#RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENTRYPOINT [ "/usr/bin/tini", "--" ]

RUN mkdir -p /model \
      && apt-get update \
      && apt-get install -y libzmq5 libzmq5-dev libsodium18 build-essential

RUN conda install -c caffe2 caffe2-cuda9.0-cudnn7=0.8.dev=py27_2018.05.14 
#RUN rm -rf /etc/apt/sources.list.d/*
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libgflags-dev \
#    libgoogle-glog-dev \
#    libgtest-dev \
#    libiomp-dev \
#    libleveldb-dev \
#    liblmdb-dev \
#    libopencv-dev \
#    libprotobuf-dev \
#    libsnappy-dev \
#    protobuf-compiler \
#    python-dev \
#    python-numpy \
#    python-pip \
#    python-pydot \
#    python-setuptools \
#    python-scipy \
#    wget \
#    && rm -rf /var/lib/apt/lists/*
#
#RUN pip install --no-cache-dir --upgrade pip setuptools wheel
#RUN pip install --no-cache-dir \
#    Cython \
#    flask \
#    future \
#    graphviz \
#    hypothesis \
#    jupyter \
#    matplotlib \
#    mock \
#    numpy>=1.13 \
#    opencv-python>=3.2 \
#    protobuf \
#    pydot \
#    python-nvd3 \
#    pyyaml>=3.12 \
#    requests \
#    scikit-image \
#    scipy \
#    setuptools \
#    six \
#    tornado
#
#
#RUN apt-get install -y --no-install-recommends libgflags-dev

#RUN cd / && git clone --recursive -b tmp https://github.com/uwsaml/nexus.git
#RUN cd /nexus && make -j$(nproc) caffe2

#RUN cd / && git clone https://github.com/icemelon9/caffe2.git && cd caffe2 && git submodule update --init --recursive \
#RUN cd / && git clone https://github.com/icemelon9/caffe2.git && cd caffe2 \
#    && mkdir build && cd build \
#    && cmake .. \
#       -DCUDA_ARCH_NAME=Manual \
#       -DCUDA_ARCH_BIN="35 52 60 61" \
#       -DCUDA_ARCH_PTX="61" \
#       -DUSE_NNPACK=OFF \
#       -DUSE_NCCL=ON \
#       -DUSE_ROCKSDB=OFF \
#       -DCMAKE_INSTALL_PREFIX=/usr/local/caffe2 \
#    && make -j"$(nproc)" install 
#RUN cd / && git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch \
#    && git submodule update --init \
#    && mkdir build \
#    && cd build \
#    && cmake ..\ 
#       -DCUDA_ARCH_NAME=Manual \
#       -DCUDA_ARCH_BIN="35 52 60 61" \
#       -DCUDA_ARCH_PTX="61" \
#       -DUSE_NNPACK=OFF \
#       -DUSE_NCCL=ON \
#       -DUSE_ROCKSDB=OFF \
#       -DUSE_MPI=OFF\
#       -DCMAKE_INSTALL_PREFIX=/usr/local/caffe2 \
#    && make install
#
#ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/caffe2/lib

RUN cd /clipper/clipper_admin/ \
    && pip install .

RUN cd / && git clone https://github.com/YuchenJin/clipper_caffe2.git
RUN conda install numpy scikit-learn requests


#COPY models/squeezenet/init_net.pb models/squeezenet/predict_net.pb /model/modules/
#COPY containers/python/caffe2_container.py containers/python/container_entry.sh /container/

#CMD ["/container/container_entry.sh", "caffe2-container", "/container/caffe2_container.py"]
