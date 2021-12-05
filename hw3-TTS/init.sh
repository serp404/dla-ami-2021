wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1/ hw_tts/data/
mv LJSpeech-1.1.tar.bz2 hw_tts/data/

git clone https://github.com/NVIDIA/waveglow.git
pip install -r requiremnts.txt
