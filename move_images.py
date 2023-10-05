import cv2
from PIL import Image
import numpy as np
import lmdb

lmdb_env = lmdb.open('datasets/CVL/h32char16to17/te')
with lmdb_env.begin(write=False) as txn:
    i = 1
    for idx,(key,val) in enumerate(txn.cursor()):
        label = key.decode()
        if('image' in label):
            img = cv2.imdecode(np.frombuffer(val,dtype=np.uint8),1)
            img= Image.fromarray(img).convert('RGB')
            image = Image.new('RGB', (16 * 13, 32), (255, 255, 255))
            image.paste(img, (0, 0))
            image.save('./imgs/cvl_ground_truth/{}.png'.format(i))
            i += 1
        #if('label' in label):
        #    with open('imgs/cvl_labels.txt', 'a') as f:
        #        f.write(val.decode())
        #        f.write('\n')