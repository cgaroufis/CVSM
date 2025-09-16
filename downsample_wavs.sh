#Script for downsampling the extracted source .wav files to 16kHz
#Usage: ./downsample_wavs.sh data_dir
#Beware! It removes the original .wav file.

for f in $1'/'*; do
    p=${f::-4}
    if [[ $f != *'down.wav' ]]; then
      echo $f
      sox $f -r 16000 $p'_down.wav'
      rm $f 
    fi
  done
done
