import cv2
import numpy as np
import argparse
import os
import pdb
import random
import tqdm
import shutil
from multiprocessing import Pool
from os.path import join

# ref: https://stackoverflow.com/questions/61130257/how-can-i-clean-this-picture-up-opencv-python
def clean_image(img,output_dir): #input size: (67,x,3)
	img_name = img.split("/")[-1]
	img = cv2.imread(img)
	# blur
	blur = cv2.GaussianBlur(img, (3,3), 0)
	# do OTSU threshold to get image
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	backtorgb = cv2.cvtColor(otsu,cv2.COLOR_GRAY2RGB)
	# write black to input image where mask is black
	cv2.imwrite(join(output_dir,img_name), backtorgb)

def rotate_image(image, angle,resizeratio):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, resizeratio)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
	return result

def noise_randombg(args,img_path,output_dir,additional_name=None): #input size: (50,50,3)
	background_img = os.listdir(args.background_dir)
	bg_img = random.choice(background_img)
	while not bg_img.endswith("jpg") and not bg_img.endswith("png"):
		bg_img = random.choice(background_img)
	if additional_name == None:
		img_name = img_path.split("/")[-1].split(".")[0]
	else:
		img_name = additional_name
	# char_name = img_name.split("_")[0]
	# output_dir = join(output_dir,char_name)
	if bg_img.endswith("jpg") or bg_img.endswith("png"):
		#--- Rotate the text img
		text_rotate_angle = random.randrange(-10, 10, 10)
		text_resize =  random.uniform(1.0, 1.2)
		img = cv2.imread(img_path)
		img = rotate_image(img,text_rotate_angle,text_resize)
		bg_img = cv2.imread(join(args.background_dir,bg_img))

		#--- flip bg images
		bg_flip = random.randint(-1,1)
		bg_img = cv2.flip(bg_img,bg_flip)
		
		#--- Resizing the bg to the shape of text image ---
		bg1 = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

		#--- Apply Otsu threshold to blue channel of the logo image ---
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		bg1_cp = bg1.copy() 
		
		#--- two color: black and blue
		color_rand = random.randint(0,1)

		#--- three channel
		channel_Blue = bg1_cp[:,:,0]
		channel_Green = bg1_cp[:,:,1]
		channel_Red = bg1_cp[:,:,2]
		three_channel = [channel_Blue,channel_Green,channel_Red]
		new_channel = []

		#---gaussian noise
		noise = np.random.normal(0, 0.1, channel_Blue.shape) *255
		#---erode kernel size
		erode_kernel_size = random.randint(2,3)
		#---Process 3 channel
		for idx,channel in enumerate(three_channel):
			for h in range(len(channel)):
				for w in range(len(channel)):
					R_G = random.randint(0,65)
					B = random.randint(50,180)
					if idx == 0:
						if otsu[h][w] == 0:
							channel[h][w] = B
					else:
						if otsu[h][w] == 0:
							channel[h][w] = R_G
			# erode
			kernel = np.ones((erode_kernel_size,erode_kernel_size), np.uint8)
			channel = cv2.erode(channel, kernel, iterations = 1)
			# add noises
			channel = channel + noise
			new_channel.append(channel)
		#---merge three channel
		bg1_cp = cv2.merge((new_channel[0],new_channel[1],new_channel[2]))
		#---gaussian blur
		bg1_cp = cv2.blur(bg1_cp,(2,2))			
		#---save images
		cv2.imwrite(join(output_dir,img_name+".jpg"), bg1_cp)

	


def main(args):
	if args.type == "donoise_3000":
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)
		dir_list =  os.listdir(args.data_dir)
		random.shuffle(dir_list)
		char_800 = set()
		with open(args.default_dict_path) as f:
			for char in f.readlines():
				char_800.add(char.strip())
		dir_list = [char for char in dir_list if char not in char_800]
		dir_list = dir_list[:3000] # 前3000個
		print("Process images...")
		pool = Pool(5,initial_pool,(args.default_dict_path,args))
		finish_list = list(tqdm.tqdm(pool.imap(process_null_3000, dir_list), total=len(dir_list)))
		pool.close()
		pool.join()	
		# move to 1000 and 1000 and 1000
		print("Move images to 3 folders...")
		dir1 = "valid_aiteam1_801"
		dir2 = "valid_aiteam2_801"
		dir3 = "valid_aiteam3_801"
		dirs = [dir1,dir2,dir3]
		if not os.path.exists(join(args.output_dir,dir1)):
			os.mkdir(join(args.output_dir,dir1))
			os.mkdir(join(args.output_dir,dir2))
			os.mkdir(join(args.output_dir,dir3))
		files_3000 = os.listdir(args.output_dir)
		random.shuffle(files_3000)
		now_foler = -1
		img_index = -1
		for img in files_3000:
			if img.endswith("jpg"):
				img_index += 1 # 不用enumerate是因為多3個folder 
				if img_index % 1000 == 0:
					now_foler+=1
				image_path = join(args.output_dir,img)
				new_foler_path = join(args.output_dir,dirs[now_foler],img)
				shutil.copy(image_path,new_foler_path)
		print("Start to zip file...")
		shutil.make_archive(dir1, 'zip', join(args.output_dir,dir1))
		shutil.make_archive(dir2, 'zip', join(args.output_dir,dir2))
		shutil.make_archive(dir3, 'zip', join(args.output_dir,dir3))
		print("Finish")
		
	elif args.type == "donoise_allnull":
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)

		dir_list =  os.listdir(args.data_dir)

		pool = Pool(5,initial_pool,(args.default_dict_path,args))
		finish_list = list(tqdm.tqdm(pool.imap(process_null_img, dir_list), total=len(dir_list)))
		pool.close()
		pool.join()	
		print(len(os.listdir(args.output_dir)))
						
def initial_pool(default_dict_path,arg):
	global char_800
	global args
	global exists_img
	args = arg
	exists_img = os.listdir(args.cache_dir)
	char_800 = set()
	with open(default_dict_path) as f:
		for char in f.readlines():
			char_800.add(char.strip())

def process_null_3000(char):
	# 每個字一張而已
	global char_800
	global exists_img
	if char not in char_800:
		curr_char_dir = join(args.data_dir,char)
		img_list = os.listdir(curr_char_dir)
		new_img_list = []
		for img in img_list:
			if img not in exists_img:
				new_img_list.append(img)
		img_list = new_img_list
		min_pic_amount = 1
		while min_pic_amount>=1:
			img_name = random.choice(img_list)
			if img_name.endswith("png"):
				min_pic_amount-= 1
				noise_randombg(args,join(curr_char_dir,img_name),args.output_dir)
	return "finish"


def process_null_img(char):
	global char_800
	global exists_img
	if char not in char_800:
		curr_char_dir = join(args.data_dir,char)
		img_list = os.listdir(curr_char_dir)
		new_img_list = []
		for img in img_list:
			if img not in exists_img:
				new_img_list.append(img)
		img_list = new_img_list
		# min_pic_amount = min(20,len(img_list))
		min_pic_amount = 20
		has_put_img = []
		repeart = 0
		for i in range(min_pic_amount):
			img_name = random.choice(img_list)
			has_put_img.append(img_name)
			if len(img_list) >5:
				img_list.remove(img_name)
			if img_name.endswith("png"):
				if img_name in has_put_img:
					additional_name = img_name.split(".")[0]+"_"+str(repeart)
					repeart+=1
				noise_randombg(args,join(curr_char_dir,img_name),args.output_dir,additional_name)
	return "finish"








if __name__ == '__main__':
	# 總共4039個class
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="../handwritten_data/clean/aiteam", type=str)
	parser.add_argument("--background_dir", default="./", type=str)
	parser.add_argument("--cache_dir", default="./aiteam_null", type=str)
	parser.add_argument("--output_dir", default="aiteam_3000/", type=str)
	parser.add_argument("--default_dict_path", default="../training data dic.txt", type=str)
	# parser.add_argument("--background2", default="./handwritten_data/dirty/background2.jpg", type=str)
	parser.add_argument("--type", default="donoise_allnull",choices=['donoise_3000',"donoise_allnull"], type=str)
	args = parser.parse_args()
	main(args)