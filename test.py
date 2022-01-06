model = keras.models.load_model('model.h5')

image_path = "image.jpg"
image_read = plt.imread(image_path)
resized_image = resize(image_read,(32,32,3))
prediction = model.predict(np.array([resized_image]))
print(class_names[np.argmax(prediction)]) #sea
