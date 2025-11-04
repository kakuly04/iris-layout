import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
 
with open("imaging/wrapped_etpu_poly_1.pkl", "rb") as f:
   entry = pickle.load(f)
 
 
cv2.imshow('check', entry["data"][0])
cv2.waitKey(0)
cv2.destroyAllWindows()