import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

#configuration variables 
DATASET_DIR=r"C:\Users\gupta\OneDrive\Desktop\embedded project\dataset"
MODELS_DIR=r"C:\Users\gupta\OneDrive\Desktop\embedded project\models"
IMG_SIZE=224
BATCH_SIZE=16
EPOCHS=15
THRESHOLD=0.5 #above this means that a human is present 
os.makedirs(MODELS_DIR,exist_ok=True)

#data loaders
#training data gets augentation to improve generalization
train_gen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    zoom_range=0.1
)

#val and test get normalized no augmentation 
val_gen=ImageDataGenerator(rescale=1./255)

train_data=train_gen.flow_from_directory(
    os.path.join(DATASET_DIR,"train"),
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data=val_gen.flow_from_directory(
    os.path.join(DATASET_DIR,"val"),
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data=val_gen.flow_from_directory(
    os.path.join(DATASET_DIR,"test"),
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class Mapping:",train_data.class_indices)

#building the model (mobilenetv2->globalaveragepooling2d->droupout(0.3)->Dense(1,"sigmoid"))

base_model=MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),
                       include_top=False,
                       weights="imagenet")

base_model.trainable=False #dont update MobileNetv2's weights during training - only train the new layers on top

x=base_model.output#storing the output of mobilenetv2 in x so i can start adding my own layers on top
x=GlobalAveragePooling2D()(x)
x=Dropout(0.3)(x)#30 percent connections switch off so that it doesnt memorize

output=Dense(1,activation="sigmoid")(x)

model=Model(inputs=base_model.input,outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",#binary cross-entropy measures the performance of a binary classfication model by quantifying the differences between predicted probabilities and actual labels
    metrics=["accuracy"]
)
model.summary()

#callbacks are funtions that run automatically during training and at the end of each epoch 
#Earlystopping stops training if val loss stops improving 
#saves time on cpu since training is slow 
early_stop=EarlyStopping(
    monitor="val_loss",
    patience=3, #stop after 3 epochs of no improvement
    restore_best_weights=True
)
#ModelCheckpoint saves the best moel automatically during training 
checkpoint=ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR,"best_classifier.h5"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1#provides a basic output or progress indicator about what a program or function is doing 
)

callbacks=[early_stop,checkpoint]

#training the model
print("\nStarting training on CPU-this will take a while...")
print("EarlyStopping will cut it short if model stops improving\n")

history=model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks
    
)
#EVALUATE ON TEST SET
print("\m Evaluating on test set....")
test_data.reset()

y_pred_probs=model.predict(test_data)
y_pred=(y_pred_probs>THRESHOLD).astype(int).flatten()
y_true=test_data.classes
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(train_data.class_indices.keys())
)
        )
#save file.h5
h5_path=os.path.join(MODELS_DIR,"human_classifier.h5")
model.save(h5_path)
print(f"\nFull model saved to {h5_path}")

# ── Convert to TFLite for Raspberry Pi ───────────────────────────
print("\nConverting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# optimise for CPU/embedded — reduces size and speeds up Pi inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = os.path.join(MODELS_DIR, "human_classifier.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_path}")

# ── Simple prediction function (used by src/classifier.py on Pi) ──
def predict(image_array):
    """
    Takes a 224x224x3 numpy array (from OpenCV frame)
    Returns True if human detected, False if not
    """
    img = image_array / 255.0                      # normalise
    img = np.expand_dims(img, axis=0)              # add batch dim
    prob = model.predict(img, verbose=0)[0][0]
    return bool(prob > THRESHOLD), float(prob)

print("\nDone. Files in models/:")
for f in os.listdir(MODELS_DIR):
    size_mb = os.path.getsize(os.path.join(MODELS_DIR, f)) / (1024*1024)
    print(f"  {f}  ({size_mb:.1f} MB)")