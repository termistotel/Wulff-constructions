import numpy as np
import matplotlib.pyplot as plt

gac = 2.02
gzz = 2.36
# gac = 1.01
# gzz = 1.01

n = 100000
picr = 440//2
dimx, dimy = 1230//2, 930//2

# print(np.arctan(np.sqrt(3) - 2*gac/gzz))
g = 1
C = 3

def grEdgeEn(fi, gac, gzz):
	tmpfi = fi%60
	tmpfi = tmpfi*np.pi/180

	en = np.zeros(fi.shape)
	en[tmpfi <= np.pi/6] = gac * np.sin(tmpfi[tmpfi <= np.pi/6]) + gzz * np.sin(np.pi/6 - tmpfi[tmpfi <= np.pi/6])
	en[tmpfi > np.pi/6] = gzz * np.sin(tmpfi[tmpfi > np.pi/6] - np.pi/6) + gac * np.sin(np.pi/6 - (tmpfi[tmpfi > np.pi/6] - np.pi/6) )

	return en

def grEdgeEn2(fi):
	tmpfi = fi%60
	tmpfi = tmpfi*np.pi/180

	en = np.zeros(fi.shape)
	en[tmpfi <= np.pi/6] = g * np.cos(tmpfi[tmpfi <= np.pi/6] + C)
	en[tmpfi > np.pi/6] = g * np.cos(np.pi/3 - tmpfi[tmpfi > np.pi/6] + C)

	return en

def main(gac, gzz):
	# Polar surface En
	fi = np.linspace(0,360, n)
	r = grEdgeEn(fi, gac, gzz)

	# Cartesius surface En
	x = r*np.cos(fi*np.pi/180)
	y = r*np.sin(fi*np.pi/180)

	# lowest energies
	# chosen = np.argsort(r)[:1000]
	chosen = np.arange(n)[(np.arange(n)%(n//100))==0]

	# plt.xlim(-min(gac,gzz), min(gac,gzz))
	# plt.ylim(-min(gac,gzz), min(gac,gzz))
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ticks = np.array([-1,-0.5,0,0.5,1])
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	plt.locator_params(axis='y', nbins=6)
	plt.locator_params(axis='x', nbins=6)
	ax.axis((-min(gac,gzz), min(gac,gzz), -min(gac,gzz), min(gac,gzz)), "scaled")

	# Wulff construction
	for i, p in zip(chosen, fi[chosen]*np.pi/180):
		t = np.linspace(-1, 1, 100)
		x1 = -1.5*np.sin(p)*t + x[i]
		y1 = 1.5*np.cos(p)*t + y[i]
		ax.plot(x1, y1, "r--", linewidth=0.5)

	ax.plot(x,y)
	plt.show()

	# Generate mask
	chosen = np.append(chosen, np.argsort(r)[:100])
	mask = np.ones(shape= (dimy, dimx) )
	xx, yy = np.meshgrid(np.arange(-dimx//2, dimx//2), np.arange(-dimy//2, dimy//2))

	for i, p in zip(chosen, fi[chosen]*np.pi/180):
		xx1 = (xx*np.cos(p) - yy*np.sin(p))
		mask[xx1 > (r[i]*picr/np.min(r)/1.5)] = 0

	plt.axis('off')
	plt.imshow(mask, cmap='gray')
	plt.show()

main(gac, 2.36)
main(gac, 2.02)
main(gac, 1.7)