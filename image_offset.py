import argparse
import os
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
import scipy.ndimage
import numpy as np

x, y, h, w  = 1220, 1020, 12, 10

def check_if_file_existed(filename):
	if os.path.isfile(filename):
		print(f'filename \t: {filename} existed')
	else:
		print(f'filename \t: {filename} is not existed')

def check_if_dir_existed(dir_name, create=False):
	if not os.path.exists(dir_name):
		print(f'folder \t\t: {dir_name} is not available')
		if create:
			os.mkdir(dir_name)
			print(f'folder \t\t: {dir_name} created')	
	else:
		print(f'folder \t\t: {dir_name} is available')

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-dataset", "--dataset", type=str, default="./dataset", required=False, help="path to the dataset")
	return vars(parser.parse_args())


def Fourier_Interpolation(image, shift):
	'''
	image is a monochrome 
	shift is an array of (x, y)
	'''
	print(f'Desired offset (y, x): {shift}')
	fourier_shifted_image = fourier_shift(np.fft.fftn(image), shift)
	return np.fft.ifftn(fourier_shifted_image)

	

def Cubic_Interpolation(image, shift):
	Mtrans = np.float32([[1,0,-shift[1]],[0,1, -shift[0]]])
	return cv2.warpAffine(image, Mtrans, image.shape[::-1], flags=cv2.INTER_CUBIC)

def Lanczos_Interpolation(image, shift):
	Mtrans = np.float32([[1,0,-shift[1]],[0,1, -shift[0]]])
	return cv2.warpAffine(image, Mtrans, image.shape[::-1], flags=cv2.INTER_LANCZOS4)

def Spline_Interpolation(image, shift):
	return scipy.ndimage.shift(image, shift)

def shift_img_along_axis( img, axis, shift, constant_values=0):
	""" 
	https://stackoverflow.com/questions/36909738/how-to-shift-image-array-with-supixel-precison-in-python
	"""
	shift = -shift
	intshift = int(shift)
	remain0 = abs( shift - int(shift) )
	remain1 = 1-remain0 #if shift is uint : remain1=1 and remain0 =0
	npad = int( np.ceil( abs( shift ) ) )  #ceil relative to 0. ( 0.5=> 1 and -0.5=> -1 )
	pad_arg = [(0,0)]*img.ndim
	pad_arg[axis] = (npad,npad)
	bigger_image = np.pad( img, pad_arg, 'constant', constant_values=constant_values) 
	
	part1 = remain1*bigger_image.take( np.arange(npad+intshift, npad+intshift+img.shape[axis]) ,axis)
	if remain0==0:
		shifted = part1
	else:
		if shift>0:
			part0 = remain0*bigger_image.take( np.arange(npad+intshift+1, npad+intshift+1+img.shape[axis]) ,axis) #
		else:
			part0 = remain0*bigger_image.take( np.arange(npad+intshift-1, npad+intshift-1+img.shape[axis]) ,axis) #

		shifted = part0 + part1
	return shifted


def sub_pixel_shift(folder_source, folder_dest, shift):
	check_if_dir_existed(folder_dest)
	for img_name in os.listdir(folder_source):
		if (img_name.endswith(".png")):
			full_img_name = f"{folder_source}/{img_name}"
			full_img_name_dest = f"{folder_dest}/s_{img_name}"
			check_if_file_existed(full_img_name)
			img_src = cv2.imread(full_img_name, cv2.COLOR_BGR2GRAY)
			img_dst = Spline_Interpolation(cv2.imread(full_img_name, cv2.COLOR_BGR2GRAY), shift)
			cv2.imwrite(full_img_name_dest, img_dst)
			
	

#def check_src_dst():
	#fig2 = plt.figure()
	#ax20 = fig2.add_subplot(1,2,1)
	#ax20.imshow(cv2.imread("./dataset/src/72_0.png", cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w])#(img_src[y:y+h, x:x+w])
	#ax20.set_title(f"Sourcsed")

	#ax20 = fig2.add_subplot(1,2,2)
	#ax20.imshow(cv2.imread("./dataset/dst/s_72_0.png", cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w])
	#ax20.set_title(f"Dest")



def main(args, shift):
	folder_source = f"{args['dataset']}/src"
	img_names = os.listdir(folder_source)
	
	# Output folder to test
	folder_dest = f"{args['dataset']}/dst"
	check_if_dir_existed(folder_dest, True)

	# Do the test of 1st image in the list	
	img_name = os.path.join(folder_source, img_names[0])
	image = cv2.imread(img_name, cv2.COLOR_BGR2GRAY)
	
	# Crop specific region just to visual test 
	
	#check_src_dst()
	sub_pixel_shift(folder_source, folder_dest, shift)
	
	shifted_image = []

	fourier_shifted_image = Fourier_Interpolation(image, shift)
	shifted_image.append(abs(fourier_shifted_image))
	shifted_image_cubic	= Cubic_Interpolation(image, shift=-shift)
	shifted_image.append(shifted_image_cubic)
	shifted_image_lanczos	= Lanczos_Interpolation(image, shift=-shift)	
	shifted_image.append(shifted_image_lanczos)
	shifted_image_spline = Spline_Interpolation(image, shift)
	shifted_image.append(shifted_image_spline)
	manual_image_shift = shift_img_along_axis(image, axis=0, shift=shift[0] )
	manual_image_shift = shift_img_along_axis(manual_image_shift, axis=1, shift=shift[1])
	shifted_image.append(manual_image_shift)

	
	fig = plt.figure()
	ax0 = fig.add_subplot(2,3,1)
	ax0.set_title(f"Sourced Image {img_name}")
	ax1 = fig.add_subplot(2,3,2)
	ax1.set_title(f"Shifted with Fourier phase shift")
	ax2 = fig.add_subplot(2,3,3)
	ax2.set_title(f"Shifted with cv2.warpAffine(...,flags=cv2.INTER_CUBIC)")
	ax3 = fig.add_subplot(2,3,4)
	ax3.set_title(f"Shifted with cv2.warpAffine(...,flags=cv2.INTER_LANCZOS4)")
	ax4 = fig.add_subplot(2,3,5)
	ax4.set_title(f"Shifted with scipy.ndimage.shift")
	ax5 = fig.add_subplot(2,3,6)
	ax5.set_title(f"Shifted with manual shift")

	ax0.imshow(image[y:y+h, x:x+w], origin='lower', cmap='gray')
	ax1.imshow(abs(fourier_shifted_image)[y:y+h, x:x+w], origin='lower', cmap='gray', interpolation='hanning')
	ax2.imshow(shifted_image_cubic[y:y+h, x:x+w], origin='lower', cmap='gray', interpolation='hanning')
	ax3.imshow(shifted_image_lanczos[y:y+h, x:x+w], origin='lower', cmap='gray', interpolation='hanning')
	ax4.imshow(shifted_image_spline[y:y+h, x:x+w], origin='lower', cmap='gray', interpolation='hanning')
	ax5.imshow(manual_image_shift[y:y+h, x:x+w], origin='lower', cmap='gray', interpolation='hanning')

	detected_shift = []
	
	for i in range(len(shifted_image)):
		s, e, dif = phase_cross_correlation(image, shifted_image[i], upsample_factor=1000)
		print(f"Detected shift {s}, Error {e}")

		detected_shift.append(s)

if __name__ == '__main__':
	args = arg_parser()
	check_if_dir_existed(args['dataset'])
	main(args, shift = np.array([3.5, 3.5 ]))
	plt.show()
	print('done')
