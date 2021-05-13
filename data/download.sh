wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GlFbs305hnh1p9vmROVWjAQKs3Zm5XZE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GlFbs305hnh1p9vmROVWjAQKs3Zm5XZE" -O dirty.zip && rm -rf /tmp/cookies.txt
unzip dirty.zip -d ./handwritten_data

if [ "$1" == "clean" ]
then
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1BMPapHnQ4LnvD_1qwEYDdFsMXHaH0_ed' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BMPapHnQ4LnvD_1qwEYDdFsMXHaH0_ed" -O clean.zip && rm -rf /tmp/cookies.txt
unzip clean.zip -d ./handwritten_data
fi