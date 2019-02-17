import numpy as np
import cv2
import matplotlib.pyplot as plt

gac = 2.02
gzzmo = 2.36
gzzs = 2.36
# gac = 1.01
# gzz = 1.01

n = 100000
picr = 440//2
dimx, dimy = 1230//2, 930//2

# print(np.arctan(np.sqrt(3) - 2*gac/gzz))
g = 1
C = 3

def grEdgeEn(fi, gac, gzzmo, gzzs):
	tmpfi = fi%120
	tmpfi = tmpfi*np.pi/180

	en = np.zeros(fi.shape)

	domain = tmpfi <= np.pi/6
	en[domain] = gac * np.sin(tmpfi[domain]) + gzzmo * np.sin(np.pi/6 - tmpfi[domain])

	domain = np.logical_and(tmpfi > np.pi/6, tmpfi <= np.pi/3)
	en[domain] = gzzs * np.sin(tmpfi[domain] - np.pi/6) + gac * np.sin(np.pi/6 - (tmpfi[domain] - np.pi/6) )

	domain = np.logical_and(tmpfi > np.pi/3, tmpfi <= np.pi/2)
	en[domain] = gac * np.sin(tmpfi[domain] - np.pi/3) + gzzs * np.sin(np.pi/6 - (tmpfi[domain] - np.pi/3) )

	domain = tmpfi > np.pi/2
	en[domain] = gzzmo * np.sin(tmpfi[domain] - np.pi/2) + gac * np.sin(np.pi/6 - (tmpfi[domain] - np.pi/2) )

	return en

def grEdgeEn2(fi):
	tmpfi = fi%60
	tmpfi = tmpfi*np.pi/180

	en = np.zeros(fi.shape)
	en[tmpfi <= np.pi/6] = g * np.cos(tmpfi[tmpfi <= np.pi/6] + C)
	en[tmpfi > np.pi/6] = g * np.cos(np.pi/3 - tmpfi[tmpfi > np.pi/6] + C)

	return en

def main(gac, gzzmo, gzzs, display=True):
	# Polar surface En
	fi = np.linspace(0,360, n)
	r = grEdgeEn(fi, gac, gzzmo, gzzs)

	# Cartesius surface En
	x = r*np.cos(fi*np.pi/180)
	y = r*np.sin(fi*np.pi/180)

	chosen = np.arange(n)[(np.arange(n)%(n//200))==0]

	if display:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ticks = np.array([-1,-0.5,0,0.5,1])
		ax.spines['left'].set_position('center')
		ax.spines['right'].set_color('none')
		ax.spines['bottom'].set_position('center')
		ax.spines['top'].set_color('none')
		plt.locator_params(axis='y', nbins=6)
		plt.locator_params(axis='x', nbins=6)
		limit = sum([gac,gzzs,gzzmo])/3
		ax.axis((-limit, limit, -limit, limit), "scaled")

		# Wulff construction
		for i, p in zip(chosen, fi[chosen]*np.pi/180):
			t = np.linspace(-limit, limit, 100)
			x1 = -2*np.sin(p)*t + x[i]
			y1 = 2*np.cos(p)*t + y[i]
			ax.plot(x1, y1, "r--", linewidth=0.5)

		ax.plot(x,y)
		plt.show()

	# Generate mask
	chosen = np.append(chosen, np.argsort(r)[:100])
	mask = np.ones(shape= (dimy, dimx) )
	xx, yy = np.meshgrid(np.arange(-dimx//2, dimx//2), np.arange(-dimy//2, dimy//2))

	for i, p in zip(chosen, fi[chosen]*np.pi/180):
		xx1 = (xx*np.cos(p) - yy*np.sin(p))
		mask[xx1 > (r[i]*picr/np.min(r)/2)] = 0

	if display:
		plt.axis('off')
		plt.imshow(mask, cmap='gray')
		plt.show()

	return mask


gac = 5
gzzmo = lambda mu: 3 - mu/3
gzzs = lambda mu: 3 + mu/3

fun = lambda gac, mu: main(gac, gzzmo(mu), gzzs(mu),  display = False)
m = 5
dimx1, dimy1 = dimx*m, dimy*m
output = np.zeros(shape=(dimy, dimx1, 3))

out = np.zeros(shape=(dimy, dimx, 3))

yellow = np.array([1, 1, 0]).reshape(1,1,3)
blue = np.array([0, 0, 1]).reshape(1,1,3)
white = np.array([1, 1, 1]).reshape(1,1,3)

acrange = np.linspace(2.9,5,m)
murange = np.linspace(0, 2, m)
colorCoef = np.linspace(0.5,1,m)
colorCoef2 = np.linspace(0.6,1,m)

for i, gac in enumerate(acrange):
	print(i)
	out = fun(gac, murange[i])

	# Get edges
	sobelx = cv2.Sobel(out,cv2.CV_64F,1,0,ksize=31)
	sobely = cv2.Sobel(out,cv2.CV_64F,0,1,ksize=31)
	edgs = np.array(np.sqrt(np.square(sobelx) + np.square(sobely)))

	out = np.array(out.reshape(*out.shape, 1))
	edgs = np.array(edgs.reshape(*edgs.shape, 1))
	edgs[out>0] = 0

	c = colorCoef[i]
	c2 = colorCoef2[i]

	out = out*white
	edgs = edgs*white > 0.5

	tmp = edgs * ((1-c)*blue + c*yellow)*c2

	# plt.imshow(tmp[:,:,0], cmap='gray')
	# plt.show()
	# print(tmp.shape, out.shape, edgs.shape)
	# print(tmp.dtype, out.dtype, edgs.dtype)

	out[edgs] = tmp[edgs]
	# out = out+ edgs * ((1-c)*blue + c*yellow + c2*white)

	output[:, i*dimx:(i+1)*dimx, :] = np.clip(out, 0, 1) 

plt.axis('off')
# plt.imshow(output, cmap='gray')
plt.imshow(output)
plt.show()

# fun(-3)
# fun(0)
# fun(3)