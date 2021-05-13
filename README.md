# Data Augmentation
分成兩種data: clean 以及 dirty
* clean: 官網的800字轉成黑白
* dirty: AIteam的4000多字轉成官網的髒背景
* 資料夾格式：
    * handwritten_data
        * clean
            * aiteam
                * 丁
                    * 丁.jpg
            * default
                * 丁
                    * clean
                        * 丁.jpg
                    * 丁.jpg
        * dirty
            * aiteam
                * 丁
                    * dirty
                        * bg1_丁.jpg
                        * bg2_丁.jpg
                    * 丁.jpg
            * default
                * 丁
                    * 丁.jpg
            * background.jpg
            * background2.jpg

## 下載資料
如果加上clean參數會連同clean一起下載
'''shell
bash download.sh clean(optonal)
'''


## decompress.py
--data_dir ./Traditional-Chinese-Handwriting-Dataset/data
--output_dir ./handwritten_data
--source 指定aiteam資料或是官網資料
'''shell
python3 decompress.py --source AITEAMorDefault
'''
## augmentation.py
--type ['donoise', 'doclean']
'''shell
python3 augmentation.py
'''

## 待處理
* 官網每個data dim目前都不相同