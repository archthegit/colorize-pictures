import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", type=str, required=True, help="Path to a black and white image")
parser.add_argument("-p", "--prototxt", type=str, required=True, help="Path to Caffe prototxt file")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to Caffe pre-trained model")
parser.add_argument("-c", "--centres", type=str, required=True, help="Path to cluster centres")

args = vars(parser.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
cluster_centres = np.load(args["centres"])

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
cluster_centres = cluster_centres.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [cluster_centres.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313], 2.606, dtype="float32")]

img = cv2.imread(args["image"])
scaled_img = img.astype("float32")/255.0
lab = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2Lab)

resized_img = cv2.resize(lab, (224,224))
L = cv2.split(resized_img)[0]
L-=50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))
ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

cv2.imwrite('./images/colorized/output.jpg', colorized)
cv2.imshow("Original", img)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
