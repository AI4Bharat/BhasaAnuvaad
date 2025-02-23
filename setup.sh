conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
python3 -m pip install "cython<3.0.0" "pyyaml==5.3.1" -y
python3 -m pip install -r requirements.txt
conda install -c conda-forge libsndfile==1.0.31 -y
conda install conda-forge::pyenchant conda-forge::enchant -y
conda install pytorch::faiss-gpu -y
