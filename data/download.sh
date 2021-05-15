wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Usxyjsy71pK-_RyCXczAciqdXSwgKcJq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Usxyjsy71pK-_RyCXczAciqdXSwgKcJq" -O dirty.zip && rm -rf /tmp/cookies.txt
unzip dirty.zip -d ./handwritten_data

if [ "$1" == "clean" ]
then
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fE4UObLUL0nkQ6ewhOwXi8Eied1LGJcU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fE4UObLUL0nkQ6ewhOwXi8Eied1LGJcU" -O clean.zip && rm -rf /tmp/cookies.txt
unzip clean.zip -d ./handwritten_data
fi