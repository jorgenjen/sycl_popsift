FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y  \
        cmake \
        libboost-all-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-program-options-dev \
        libboost-thread-dev \
        libdevil-dev \
        zsh \
        fzf \
        python3.12 \
        python3.12-venv \
        python3-pip \
        unzip \
        curl \
        wget \
        && rm -rf /var/lib/apt/lists/* 

# Download and install Neovim
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz \
    && tar -C /opt -xzf nvim-linux-x86_64.tar.gz \
    && rm nvim-linux-x86_64.tar.gz

ENV PATH="/opt/nvim-linux-x86_64/bin:${PATH}"

RUN useradd -m nvim 

RUN mkdir -p ~/.config 

RUN cd $HOME && git clone https://github.com/jorgenjen/dotfiles.git && echo "mr firisk"
RUN cd $HOME/dotfiles/ && git checkout sycl_docker && bash $HOME/dotfiles/link.sh nvim

RUN nvim --headless "+Lazy! sync" +qa
RUN nvim --headless -c "MasonUpdate" -c "q"
RUN nvim --headless -c "MasonInstall clangd clang-format cmakelang" -c "q"
RUN nvim --headless -c "TSInstallSync cpp c" -c "q" # Seems like treesitter is not working as it should

RUN mkdir -p /home/nvim/personal # Mount in this one

WORKDIR /home/nvim/personal/
CMD ["/bin/bash"]
