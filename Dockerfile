FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu24.04


RUN apt-get update && apt-get install -y  \
        cmake \
        libboost-all-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-program-options-dev \
        libboost-thread-dev \
        libdevil-dev \
        neovim \
        zsh \
        fzf \
        librust-starship-module-config-derive-dev \
        zoxide \
        python3.12-minimal \
        unzip \
        && rm -rf /var/lib/apt/lists/* # To reduce package size


# ARG USERNAME=nvim


RUN useradd -m nvim 
# USER nvim # Don't set continue to be root so that we can edit filesycl                                                    latest                         7d9c7c709781   About a minute ago   15.1GBs


# Ensure .config directory exists
RUN mkdir -p ~/.config 

RUN cd $HOME && git clone https://github.com/jorgenjen/dotfiles.git 

RUN cd $HOME/dotfiles/ && git checkout sycl_docker && bash ~/dotfiles/link.sh nvim

# RUN cd $HOME/ && nvim --headless --cmd "call lazy#sync()"
# RUN yes | nvim --headless --cmd 'call lazy#sync()'


RUN nvim --headless "+Lazy! sync" +qa

RUN mkdir -p $HOME/personal # Mount in this one


# # Set default shell to bash
CMD ["/bin/bash"]
