# Upsamples wav files in a given directory, and then extracts target sources
# Place it in the target directory!
# Usage: ./get_stems.sh source_data_dir

for d in $1'/'*; do
  umx $d --targets 'vocals' --niter 0
  f=${d##*/}
  g=${f::-4}'_umxl/vocals.wav'
  echo $g
  p=${g::-4}
  sox $g -r 16000 ${f::-4}'_down.wav'
  rm -rf ${f::-4}'_umxl/'
done
