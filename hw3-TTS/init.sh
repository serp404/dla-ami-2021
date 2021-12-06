pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
mkdir hw_tts/data
mv LJSpeech-1.1/ hw_tts/data/
mv LJSpeech-1.1.tar.bz2 hw_tts/data/
git clone https://github.com/NVIDIA/waveglow.git
pip install -r requirements.txt
gdown --id 1sOoj2OH7tWvgTqHtQKt_E3t2KVO_Mo23
