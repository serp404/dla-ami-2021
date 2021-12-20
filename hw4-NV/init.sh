pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
mkdir hw_nv/data/
mv LJSpeech-1.1 hw_nv/data/
mv LJSpeech-1.1.tar.bz2 hw_nv/data/
mkdir hw_nv/log/
pip install -r requirements.txt
gdown --id 1rk_3vtLtDzwCmj0wfsr99FhR6wlekR_r
