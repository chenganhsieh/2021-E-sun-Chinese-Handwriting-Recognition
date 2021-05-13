import cv2
import numpy as np
import argparse
import os
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

def noise_image(args,img,output_dir): #input size: (50,50,3)
	# print(img)
	img_name = img.split("/")[-1]
	img = cv2.imread(img)
	text_rotate_angle = random.randrange(-20, 20, 10)
	text_resize =  random.uniform(0.6, 0.8)
	img = rotate_image(img,text_rotate_angle,text_resize)

	bg_img1 = cv2.imread(args.background1)
	bg_img2 = cv2.imread(args.background2)

	# bg1_rotate_angle =  random.randrange(-20, 20, 10)
	# bg2_rotate_angle =  random.randrange(-20, 20, 10)
	bg1_flip = random.randint(-1,1)
	bg2_flip = random.randint(-1,1)
	bg_img1 = cv2.flip(bg_img1,bg1_flip)
	# bg_img1 = rotate_image(bg_img1,bg1_rotate_angle)
	bg_img2 = cv2.flip(bg_img2,bg2_flip)
	# bg_img2 = rotate_image(bg_img2,bg2_rotate_angle)

	#--- Resizing the bg to the shape of text image ---
	bg1 = cv2.resize(bg_img1, (img.shape[1], img.shape[0]))
	bg2 = cv2.resize(bg_img2, (img.shape[1], img.shape[0]))

	#--- Apply Otsu threshold to blue channel of the logo image ---
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	# text_rotate_angle =  random.randrange(10, 20, 10)
	# text_rotate_angle = 5
	# text_resize =  random.uniform(0.8, 0.9)
	# otsu = rotate_image(otsu,text_rotate_angle,text_resize)
	# otsu = cv2.resize(otsu,(int(otsu.shape[1]*text_resize),int(otsu.shape[0]*text_resize)))



	# ret, text_mask = cv2.threshold(img[:,:,0], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	bg1_cp = bg1.copy() 
	bg2_cp = bg2.copy()
	#--- Copy pixel values of logo image to room image wherever the mask is white ---
	# bg1_cp[np.where(otsu == 0)] = img[np.where(otsu == 0)]
	# bg2_cp[np.where(otsu == 0)] = img[np.where(otsu == 0)]
	bg1_cp[np.where(otsu == 0)] = 0
	bg2_cp[np.where(otsu == 0)] = 0
	cv2.imwrite(join(output_dir,"bg1_"+img_name), bg1_cp)
	cv2.imwrite(join(output_dir,"bg2_"+img_name), bg2_cp)

	

	






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
	else:
		data_path = join(args.data_dir,'dirty/aiteam')
		dir_list =  os.listdir(data_path)
		for word in dir_list:
			if len(word) == 1:
				print(f"\nProcessing word {word}:")
				word_dir = join(data_path,word)
				file_list =  os.listdir(word_dir)
				if not os.path.exists(join(word_dir,'dirty')):
					os.makedirs(join(word_dir,'dirty'), exist_ok=True)
				for image in file_list:
					if image.endswith('.png'):
						print(f"Processing image {image}",end='\r')
						noise_image(args,join(word_dir,image),join(word_dir,'dirty'))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="./handwritten_data", type=str)
	parser.add_argument("--background1", default="./handwritten_data/dirty/background.jpg", type=str)
	parser.add_argument("--background2", default="./handwritten_data/dirty/background2.jpg", type=str)
	parser.add_argument("--type", default="doclean",choices=['donoise', 'doclean'], type=str)
	args = parser.parse_args()
	main(args)