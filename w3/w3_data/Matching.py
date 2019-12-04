import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import cv2
import time
import os
def meanPooling(img, level):
	row, col = img.shape
	row_ = row//level
	col_ = col//level
	imgmat = np.array(img)
	imgout = np.zeros((row_, col_))
	for i in range(row_):
		for j in range(col_):
			imgout[i,j] = np.mean(imgmat[i*level:(i+1)*level, j*level:(j+1)*level])
	return imgout

def correctioncompute(img,temp, command):
	if command=="SSD":
		correctionMax = 100000000  # 最大相关值
	elif command=="corr":
		correctionMax = 0
	else:
		return False
	lx = ly = 0  # 最大相关值坐标
	row, col = img.shape
	row_, col_ = temp.shape
	arrimg = np.array(img)
	arrtemp = np.array(temp)

	deeparr = np.zeros((col - col_,row-row_))
	for x in range(col - col_):
		for y in range(row - row_):
			M = arrimg[y:y+row_, x:x+col_].astype('float64')
			N = arrtemp.astype('float64')
			if command == "corr":
				corr = np.sum(M * N).astype('float64') / \
					   np.sqrt(np.sum(M ** 2).astype('float64') * np.sum(
						   N ** 2).astype('float64'))
				deeparr[x, y] = corr
				if corr > correctionMax:
					correctionMax = corr
					lx = x
					ly = y
			elif command == "SSD":
				corr = np.sum((M-N)**2)
				if corr < correctionMax:
					correctionMax = corr
					lx = x
					ly = y
			deeparr[x, y] = corr

	return corr, lx, ly, deeparr

def correctionMatching(simg, timg, command):
	level = 0  # 图像缩放尺度幂次
	corre, x, y, deeparr = correctioncompute(simg, timg, command)

	print(level, corre, x, simg.shape[0]-(y+timg.shape[0]))
	plt.figure()
	plt.imshow(simg,cmap='gray')
	currentFigure = plt.gca()
	rect = patches.Rectangle((x,y), timg.shape[1], timg.shape[0],
							 linewidth=1, edgecolor='r', facecolor='none')
	currentFigure.add_patch(rect)
	plt.scatter(x, y+timg.shape[0],marker = 'o', s=80, cmap=plt.cm.Spectral)

	fig = plt.figure()
	deep = fig.gca(projection='3d')
	X = np.arange(np.shape(deeparr)[1])
	Y = np.arange(np.shape(deeparr)[0])
	X, Y = np.meshgrid(X, Y)
	# Z = np.sin(np.sqrt(X**2+Y**2))
	Z = deeparr
	surf = deep.plot_surface(X, Y, Z, cmap=cm.jet,linewidth=0, antialiased=False)
	deep.set_zlim(deeparr.min(),deeparr.max())
	deep.zaxis.set_major_locator(LinearLocator(10))
	# deep.zaxis.set_major_locator(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

def hausforffDistance(simg, timg):
	row, col = simg.shape
	row_, col_ = timg.shape
	lxT, lyT = np.where(timg == 255)
	maxnum = lx = ly =0
	for x in range(col - col_):
		for y in range(row - row_):
			print(time.ctime())
			Barr = np.array(simg[y:y + row_, x:x + col_])
			lxA, lyA = np.where(Barr == 255)

			maxA = []
			for a in range(len(lxA)):  		# a in simg
				minB = []
				for b in range(len(lxT)):		# b in timg
					minB.append((lxA[a]-lxT[b])**2+(lyA[a]-lyT[b])**2)
				maxA.append(min(minB))
			maxHAB = max(maxA)

			maxB = []
			for b in range(len(lxT)):  # b in timg
				minA = []
				for a in range(len(lxA)):  # a in simg
					minA.append((lxA[a] - lxT[b])**2+(lyA[a] - lyT[b])**2)
				maxB.append(min(minA))
			maxHBA = max(maxB)

			if max(maxHAB, maxHBA) > maxnum:
				maxnum = max(maxHAB, maxHBA)
				lx = x
				ly = y

	print(maxnum, lx, ly)
	plt.figure()
	plt.imshow(simg, cmap='gray')
	currentFigure = plt.gca()
	rect = patches.Rectangle((x, y), timg.shape[1], timg.shape[0],
							 linewidth=1, edgecolor='r', facecolor='none')
	currentFigure.add_patch(rect)
	plt.scatter(x, y, marker='o', s=80, cmap=plt.cm.Spectral)
	plt.show()

if __name__=="__main__":
	scene = plt.imread("./Scene.jpg")
	print(os.getcwd())
	template1 = plt.imread("./Template_1.jpg")
	template2 = plt.imread("./Template_2.jpg")

	# 相关性匹配
	# correctionMatching(scene, template1, "corr")
	# correctionMatching(scene, template2, "corr")
	# correctionMatching(scene, template1, "SSD")
	# correctionMatching(scene, template2, "SSD")
	# canny
	scenefilter = cv2.GaussianBlur(scene,(5,5),0)
	temp1filter = cv2.GaussianBlur(template1,(5,5),0)
	temp2filter = cv2.GaussianBlur(template2, (5, 5), 0)
	scenecan = cv2.Canny(scenefilter, 45, 135)
	temp1can = cv2.Canny(temp1filter, 105, 225)
	temp2can = cv2.Canny(temp2filter, 45, 135)

	# hausforffDistance
	print(time.ctime())
	maxvalue, lx, ly = hausforffDistance(scenecan, temp1can)
	print(time.ctime())