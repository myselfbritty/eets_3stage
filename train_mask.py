import tensorflow as tf
from mask_classifier import make_classifier
import numpy as np
import os
from sys import exit
import cv2
from sklearn.utils import shuffle

model = make_classifier(num_classes=10)

base_train_dir = '/home/dthapar/workspace/s3net/eets/train'

train_images=[]
train_mask=[]
train_label=[]
train_labels=[]

for instrument in range(10):
    current_class  = os.path.join(base_train_dir,str(instrument))
    for data_point in os.listdir(current_class):
        current_data_path = os.path.join(current_class,data_point)
        current_data=np.load(current_data_path)
        current_image = current_data[:,:,:3]
        current_mask = current_data[:,:,3]*255
        current_image = np.transpose(current_image, [2,0,1])
        train_images.append(current_image)
        current_mask = cv2.resize(current_mask,(56,56))
        current_mask = np.expand_dims(current_mask, axis = 0)
        train_mask.append(current_mask/255.0)
        current_label = np.zeros((10,))
        current_label[instrument] = 1
        train_label.append(current_label)
        train_labels.append(instrument)

train_images = np.asarray(train_images)
train_label = np.asarray(train_label)
train_labels = np.asarray(train_labels)
train_mask = np.asarray(train_mask)

print(train_images.shape)
print(train_label.shape)
print(train_mask.shape)


########################################################################################################################
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
@tf.function
def train_step(image, mask, label):
    with tf.GradientTape() as tape:
        op = model([image, mask, label], mode='softmax', training = True)
        loss = loss_fn(label,op)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss





batch_size = 32
num_of_epochs = 100
num_of_batches = train_images.shape[0]//batch_size
# num_of_val_batches = val_images.shape[0]//batch_size
# val_count = 1
# best_val_loss = 1000.0
for epoch in range(num_of_epochs):
    this_epoch_loss = 0.0
    train_images, train_label, train_mask = shuffle(train_images, train_label, train_mask, random_state=666)
    for i in range(num_of_batches):
        image_batch = train_images[i*batch_size:i*batch_size+batch_size,:,:,:]
        label_batch = train_label[i*batch_size:i*batch_size+batch_size]
        mask_batch = train_mask[i*batch_size:i*batch_size+batch_size]

        loss = train_step(image_batch, mask_batch, label_batch)
        # print('Epoch '+str(epoch)+'\tBatch '+str(i)+'\tLoss: '+str(loss)+'\n')
        this_epoch_loss+=loss.numpy()
        
        # if (i+1)%50 == 0:
        #     this_val_loss = 0.0
        #     for j in range(num_of_val_batches):
        #         image_batch = val_images[j*batch_size:j*batch_size+batch_size,:,:,:]
        #         label_batch = val_labels[j*batch_size:j*batch_size+batch_size]
        #         mask_batch = val_mask[j*batch_size:j*batch_size+batch_size]
        #         mask_gt_batch = val_mask_gt[j*batch_size:j*batch_size+batch_size]
                
        #         loss = val_step(image_batch, mask_batch, label_batch, mask_gt_batch)
                
        #         loss = loss.numpy()
        #         this_val_loss += loss
        #     this_val_loss /= num_of_val_batches
        #     print('Validation: '+str(val_count) + '\tLoss: '+str(this_val_loss))
        #     val_count += 1
        #     with open(val_loss_file_path,'a') as f:
        #         f.write(str(this_val_loss)+'\n')
        #     if this_val_loss <= best_val_loss:
        #         print('Model Loss improved from: ' + str(best_val_loss) + 'to: ' + str(this_val_loss) + '\nSaving Weights\n')
        #         best_val_loss = this_val_loss
        #         model.save_weights(model_save_path)
                
    model.save_weights('./pre-trained-weights/Stage_3/maskrcnn.h5')
    this_epoch_loss /= num_of_batches
    print('Epoch '+str(epoch)+'\tLoss: '+str(this_epoch_loss)+'\n')
    # with open(loss_file_path,'a') as f:
    #     f.write(str(this_epoch_loss)+'\n')


