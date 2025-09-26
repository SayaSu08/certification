import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

model=models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(
    x_train,y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test,y_test),
    verbose=1
)

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
print("Test Accuracy;",round(test_acc*100,2),"%")

prediction = model.predict(x_test[:1])
predicted_label=prediction.argmax()

plt.imshow(x_test[0].reshape(28,28),cmap="gray")
plt.title("prediction:"+str(predicted_label))
plt.axis("off")
plt.show()