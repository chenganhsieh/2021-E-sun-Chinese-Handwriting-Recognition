import cv2
import numpy as np
import argparse
import os
import pdb
import random
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

def noise_image(args,img_path,output_dir): #input size: (50,50,3)
	background_img = os.listdir(args.background_dir)
	img_name = img_path.split("/")[-1].split(".")[0]
	idx = 0
	for bg_img in background_img:
		if bg_img.endswith("jpg") or bg_img.endswith("png"):
			idx+=1
			# print(img)
			text_rotate_angle = random.randrange(-20, 20, 10)
			text_resize =  random.uniform(0.7, 1.0)
			img = cv2.imread(img_path)
			img = rotate_image(img,text_rotate_angle,text_resize)
			bg_img = cv2.imread(join(args.background_dir,bg_img))
			
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
			if color_rand == 0:
				black = random.randint(0,50)
				bg1_cp[np.where(otsu == 0)] = 0
			elif color_rand == 1:
				blue = random.randint(90,145) # different color shade
				for h in range(len(bg1_cp)):
					for w in range(len(bg1_cp[h])):
						for i,channel in enumerate(bg1_cp[h][w]):
							if i == 0:
								if otsu[h][w] == 0:
									bg1_cp[h][w][i] = blue
							else:
								if otsu[h][w] == 0:
									bg1_cp[h][w][i] = 0
			cv2.imwrite(join(output_dir,img_name+"_bg"+str(idx)+".png"), bg1_cp)

	


def main(args):
	if args.type == "doclean":
		data_path = join(args.data_dir,'clean/default')
		dir_list =  os.listdir(data_path)
		for word in dir_list:
			print(f"\nProcessing word {word}:")
			word_dir = join(data_path,word)
			file_list =  os.listdir(word_dir)
			if not os.path.exists(join(word_dir,'clean')):
				os.makedirs(join(word_dir,'clean'), exist_ok=True)
			for image in file_list:
				if image.endswith('.jpg'):
					print(f"Processing image {image}",end='\r')
					clean_image(join(word_dir,image),join(word_dir,'clean'))
	elif args.type == "donoise_null":
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)
		args.output_dir = join(args.output_dir,"aiteam_null")
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)
		# data_path = join(args.data_dir,'dirty/aiteam')
		dir_list =  os.listdir(args.data_dir)
		for img in dir_list:
			if img.endswith("png"):
				print(f"\nProcessing img {img}:")
				noise_image(args,join(args.data_dir,img),args.output_dir)
	elif args.type == "donoise_default":
		with open(args.default_dict_path) as f:
			for char in f.readlines():
				char = char.strip()
				if not os.path.exists(join(args.output_dir,char)):
					os.mkdir(join(args.output_dir,char))
				temp_output_dir = join(args.output_dir,char)
				if os.path.exists(join(args.data_dir,char)):
					temp_dir = join(args.data_dir,char)
					dir_list = os.listdir(temp_dir)
					for img in dir_list:
						if img.endswith("png"):
							print(f"\nProcessing img {img}:")
							noise_image(args,join(temp_dir,img),temp_output_dir)






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="../handwritten_data/clean/aiteam", type=str)
	parser.add_argument("--background_dir", default="./", type=str)
	parser.add_argument("--output_dir", default="output/", type=str)
	parser.add_argument("--default_dict_path", default="../training data dic.txt", type=str)
	# parser.add_argument("--background2", default="./handwritten_data/dirty/background2.jpg", type=str)
	parser.add_argument("--type", default="donoise_default",choices=['donoise_default','donoise_null', 'doclean'], type=str)
	args = parser.parse_args()
	

	main(args)