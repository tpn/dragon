FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

COPY zscaler-root-ca.cr[t] /usr/local/share/ca-certificates/zscaler-root-ca.crt

# This is frequently needed by uv if you use it
ENV SSL_CLIENT_CERT=/usr/local/share/ca-certificates/zscaler-root-ca.crt

# Install SSH server
RUN apt-get update \
    && apt-get install -y \
    ca-certificates git \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    vim-tiny sudo openssh-server \
    && mkdir /var/run/sshd \
    && rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

# Download pyenv to install Python
ENV PYENV_ROOT=/opt/pyenv
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT

# Compile dynamic bash extension to speed up pyenv
RUN cd $PYENV_ROOT && src/configure && make -C src

# Build Python 3.12.12 by default.
ARG python_version=3.12.12
# Keep Python source files in /usr/local/src
ENV PYTHON_BUILD_BUILD_PATH=/usr/local/src/
# Install a Python with a shared object
ENV PYTHON_CONFIGURE_OPTS="--enable-shared"
# Build Python and clean-up downloaded tarball
RUN /opt/pyenv/plugins/python-build/bin/python-build --verbose --keep ${python_version} /usr/local \
    && rm -fr /usr/local/src/Python-${python_version}.tar.*

# Create a non-root 'vscode' user
RUN groupadd -r vscode && useradd -rm -s /bin/bash -g vscode vscode
RUN echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Configure authorized_keys for the 'vscode' user
RUN mkdir -p /home/vscode/.ssh && chmod 700 /home/vscode/.ssh
COPY id_ed25519.pub /home/vscode/.ssh/authorized_keys
RUN chown -R vscode:vscode /home/vscode/.ssh

EXPOSE 22
WORKDIR /workspace
USER vscode