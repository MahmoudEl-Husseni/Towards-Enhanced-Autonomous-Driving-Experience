import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('/content/my_model_new.h5')

class_labels = [
  'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
  'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
  'End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)',
  'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
  'Priority road', 'Yield', 'Stop', 'No vehicles','Veh > 3.5 tons prohibited',
  'No entry', 'General caution', 'Dangerous curve left','Dangerous curve right',
  'Double curve', 'Bumpy road', 'Slippery road','Road narrows on the right',
  'Road work', 'Traffic signals', 'Pedestrians','Children crossing',
  'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
  'End speed + passing limits', 'Turn right ahead', 'Turn left ahead',
  'Ahead only','Go straight or right', 'Go straight or left', 'Keep right',
  'Keep left','Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons'
]

img = cv2.imread('speed_limit_100.jpg')
resized_img = cv2.resize(img, (144, 144))
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
idx = 99
img = np.expand_dims(resized_img, axis=0)
predictions = model.predict(img)
class_idx = np.argmax(predictions, axis=-1)
print(class_labels[class_idx[0]])
print(predictions[0][class_idx[0]])