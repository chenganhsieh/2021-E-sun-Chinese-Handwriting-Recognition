# Data Augmentation
分成兩種data: clean 以及 dirty
* clean: 官網的800字轉成黑白 444.4MB
* dirty: AIteam的4000多字轉成官網的髒背景 2.96GB
* 資料夾格式：
    * handwritten_data
        * clean
            * aiteam
                * 丁
                    * 丁.jpg
            * default
                * 丁
                    * 丁.jpg
        * dirty
            * aiteam
                * 丁
                    * bg1_丁.jpg
                    * bg2_丁.jpg
            * default
                * 丁
                    * 丁.jpg
            * background.jpg
            * background2.jpg

## 下載資料
如果加上clean參數會連同clean一起下載
```shell
bash download.sh clean(optonal)
```


## decompress.py
--data_dir ./Traditional-Chinese-Handwriting-Dataset/data
--output_dir ./handwritten_data
--source 指定aiteam資料或是官網資料
```shell
python3 decompress.py --source AITEAMorDefault
```
## augmentation.py
--type ['donoise', 'doclean']
```shell
python3 augmentation.py
```

## 待處理
* 官網每個data dim目前都不固定


---
# Data Cleaning

## 原先官網的60000多張圖片，按照label存成資料夾
```
# 原先官網的60000多張圖片，按照label存成資料夾
gdown --id '1hQ42JgyosLSzryex4hNPKntTkGytIh1B' --output data.zip
# 加入 AIteam 資料並有10個 background
gdown --id '1XRbwGpWWkvTVvmA0u5HdjbYXvUaaVdds' --output aiteam_10background.zip
```

## TODO: 
每個人要clean的資料，除了 廖威仲是split_1之外，其餘的split_2/3/4 各由一個人認領
```
gdown --id '1w5aPcqMebeqSN8w2ehvPJG8MAZJTGIS8' --output cleaning.zip
```
