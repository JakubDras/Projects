import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
print("Path to dataset files:", path)
destination = '/content/DATA1'

new_path = os.path.join(destination, os.path.basename(path))
shutil.move(path, new_path)


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


classes = ['angry','fearful','happy','sad','surprised','neutral']
df = pd.DataFrame()
img = image.load_img('/content/DATA1/1/test/angry/im0.png', color_mode='grayscale')
x = image.img_to_array(img)
x = x/255
x = np.expand_dims(x, axis=0)
i = 0
for batch in datagen.flow(x, batch_size=1):
  batch = batch.reshape(1,2304)
  batch = pd.DataFrame(batch)
  batch['emotion'] = "angry"
  df = pd.concat([df, batch], ignore_index=True)
  i += 1
  if i > 3:
      break

w = df.iloc[3]
w = w[:-1]
w = w.astype(float)
w = np.reshape(w, (48,48))
plt.imshow(w, cmap='gray')
plt.show()

