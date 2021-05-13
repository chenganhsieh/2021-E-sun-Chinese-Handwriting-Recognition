
import os
import zipfile
import shutil
import pdb
import argparse
from os.path import join


def decompressfile(args):
    CompressedFiles = []
    if args.source == "aiteam":
        for item in os.listdir(args.data_dir):  
            if item.endswith('.zip'): # Check for ".zip" extension.
                file_path = os.path.join(args.data_dir,item)
                CompressedFiles.append(file_path)

        for zip_file in CompressedFiles:    
            print(f"Decompress file {zip_file} now...") 
            # Construct a ZipFile object with the filename, and then extract it.
            zip_ref = zipfile.ZipFile(zip_file).extractall(args.output_dir) 
            
            source_path_dirty = args.dirtydata_dir + '/aiteam'
            source_path_clean = args.cleandata_dir + '/aiteam'
            source_path = args.output_dir + '/cleaned_data(50_50)'

            img_list = os.listdir(source_path)
            # Move a file to two location- clean and dirty.(copy first)
            for img in img_list:
                shutil.copy(source_path + '/' + img, source_path_dirty) 
                shutil.move(source_path + '/' + img, source_path_clean) 
            
            shutil.rmtree(args.output_dir + '/cleaned_data(50_50)') 
            print(f'Decompress successfully {zip_file} ......')
    else:
        pass

def organizefiles(args):
    print( 'Moving images according to traditional Chinese characters......' )
    if args.source == "aiteam":
        clean_data_path = join(args.cleandata_dir,args.source)
        dirty_data_path = join(args.dirtydata_dir,args.source)
        # clean and dirty image name should be the same
        ImageList = os.listdir(clean_data_path)
        ImageList = [img for img in ImageList if len(img)>1]
        WordList = list(set([w.split('_')[0] for w in ImageList]))

        for w in WordList:
            print(f"deal with word:{w}",end="\r")
            try:
                # Create the new word folder in OutputPath.
                os.mkdir(join(clean_data_path,w)) 
                os.mkdir(join(dirty_data_path,w)) 
                MoveList = [img for img in ImageList if w in img]
            finally:            
                for img in MoveList:
                    old_clean_path = clean_data_path + '/' + img
                    new_clean_path = clean_data_path + '/' + w + '/' + img

                    old_noise_path = dirty_data_path + '/' + img
                    new_noise_path = dirty_data_path + '/' + w + '/' + img
                    shutil.move( old_clean_path, new_clean_path )
                    shutil.move( old_noise_path, new_noise_path )
        print("Finish moving")
    else:
        clean_data_path = join(args.cleandata_dir,args.source)
        dirty_data_path = join(args.dirtydata_dir,args.source)
        ImageList = os.listdir(args.data_dir)
        ImageList = [img for img in ImageList if img.endswith('jpg')]
        WordList = set()
        for w in ImageList:
            if '_' in w:
                WordList.add(w.split('_')[1].split('.')[0])
            else:
                WordList.add(w.split('.')[0])
        WordList = list(WordList)
        for w in WordList:
            print(f"deal with word:{w}",end="\r")
            try:
                # Create the new word folder in OutputPath.
                os.mkdir(join(clean_data_path,w)) 
                os.mkdir(join(dirty_data_path,w)) 
                MoveList = [img for img in ImageList if w in img]
            finally:            
                for img in MoveList:
                    old_clean_path = args.data_dir + '/' + img
                    new_clean_path = clean_data_path + '/' + w + '/' + img

                    old_noise_path = args.data_dir + '/' + img
                    new_noise_path = dirty_data_path + '/' + w + '/' + img
                    shutil.copy( old_clean_path, new_clean_path )
                    shutil.copy( old_noise_path, new_noise_path )
        print("Finish moving")


def main(args):
    if args.source == "aiteam":
        decompressfile(args)
        organizefiles(args)
    else:
        organizefiles(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Traditional-Chinese-Handwriting-Dataset/data", type=str)
    parser.add_argument("--output_dir", default="./handwritten_data", type=str)
    parser.add_argument("--cleandata_dir", default="./handwritten_data/clean", type=str)
    parser.add_argument("--dirtydata_dir", default="./handwritten_data/dirty", type=str)
    parser.add_argument("--source", default="aiteam",choices=['aiteam', 'default'], type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.cleandata_dir):
        os.makedirs(args.cleandata_dir, exist_ok=True)
    if not os.path.exists(args.dirtydata_dir):
        os.makedirs(args.dirtydata_dir, exist_ok=True)
    if not os.path.exists(join(args.cleandata_dir,args.source)):
        os.makedirs(join(args.cleandata_dir,args.source), exist_ok=True)
    if not os.path.exists(join(args.dirtydata_dir,args.source)):
        os.makedirs(join(args.dirtydata_dir,args.source), exist_ok=True)

    main(args)