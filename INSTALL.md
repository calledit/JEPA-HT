# install

- minicoda
- git

### Windows

conda create -n jepaht python=3.11 -y

conda activate jepaht

conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install xformers==0.0.29.post1 --index-url https://download.pytorch.org/whl/cu124
pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2%2Bcu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl

pip install numpy matplotlib pandas scipy scikit-learn umap-learn datasets
