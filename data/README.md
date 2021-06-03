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
# 純 AIteam
gdown --id '1QptjrrnYIQ5oV_QuDQi0NPBTLuha65Gr' --output pure_aiteam.zip
# 純 玉山(clean)
gdown --id '17E-lVOhtKhmH20Qa2MR_TwJ_s5zmRMnR' --output pure_ysun.zip
# 玉山-1 (clean)
gdown --id '10IhtAVjGoj4kWxaciDMnjjXuRm41YnS3' --output ysun-1.zip
# 玉山-2 (clean)
gdown --id '1--MRDZxuK7xYjn9ya13kWHE85KuWZqQA' --output ysun-2.zip
# 玉山-3 (clean)
gdown --id '1-777mw8Y-Fzg7_v_lqImt8ch6moCC001' --output ysun-3.zip
# 玉山-isnull (clean)
gdown --id '1t7IBN0gPXLq1rS3X988kvw4iiGVi4xLQ' --output isnull.zip
# 玉山-isnull_large (clean)
gdown --id '1HF9wr7LnnSiYvgk6y-JD28qPsYpzaX1H' --output isnull_large.zip


```
## 6/1 更新 by正安
```
- 801 setting (w/ isnull)
#valid (aiteam)
gdown --id "1xxxFtF9hXJKl3VWO3mNLez5yKvidfywI" --output valid_aiteam1_801.zip
gdown --id "1bxknKAhploHRQ9Xxex9WFPO7phdECEdH" --output valid_aiteam2_801.zip
gdown --id "1GeGsyXBX0Sq3FV46j0l4WeGx5uusDnlH" --output valid_aiteam3_801.zip

- 4000 setting (w/o isnull -> 總共4839 class in folder)
# train - (aitem)
gdown --id "1TPnSn-OCQJ2Lvka38YQwm7oPM6loREPT" --output aiteam_80000.zip
```


## 5/31 更新 by威仲
```
- 801 setting (w/ isnull)
#train - (aiteam)
gdown --id "1-6zGHr5L6wLsSD5zdc77IH3RCbepLRRo" --output train_801.zip
#valid (ysun 3fold)
gdown --id "1-7yldjTMDldegcUtBbHnQFxixMwKEzUT" --output valid_ysun1_801.zip
gdown --id "1-E0S6G2gXz_A9z9zZZxe5dlqCYPxMwj4" --output valid_ysun2_801.zip
gdown --id "1-QppMmKphDQkelpT9LKhCYXHzNU5TqVc" --output valid_ysun3_801.zip

- 4000 setting (w/o isnull -> 總共4839 class in folder)
# train - (aitem)
gdown --id "1-R3yb-ZLLUwXx7c6Q7YN37KhdY3FE_3Y" --output train_4000.zip
#valid (ysun 3fold)
gdown --id "1-UTGFeVjSOkzxwJZQ6YDt3zZP9jxLnKK" --output valid_ysun1_4000.zip
gdown --id "1-UXgXzi3mUb1nW1q9TkTlkmiXZ3Z0Ec0" --output valid_ysun2_4000.zip
gdown --id "1-Ylg4TYFZUw468bgJZKUtscwYZevwyh0" --output valid_ysun3_4000.zip



- 4000 new setting (with 正安給的新的83000資料)
# train - (aitem)
gdown --id "14_-zwqBYDUTXqrzX0CWv2wlbtGwD4-Xf" --output train_4000.zip
#valid (ysun 3fold)
gdown --id "1-EaRBXZGuoLN4jkYkIZ9bzvgq2oyqXqA" --output valid_ysun1_4000.zip
gdown --id "1-7zFYFarh0B6nBnbbTwbgZVrcDYhgfCD" --output valid_ysun2_4000.zip
gdown --id "1-4cNHKKJBHXRTqt6_lmJez-LepGlWnys" --output valid_ysun3_4000.zip
```



## TODO: 
每個人要clean的資料，除了 廖威仲是split_1之外，其餘的split_2/3/4 各由一個人認領
```
gdown --id '1w5aPcqMebeqSN8w2ehvPJG8MAZJTGIS8' --output cleaning.zip
```
