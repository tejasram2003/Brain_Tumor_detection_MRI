from django.shortcuts import render
from django.http.response import JsonResponse
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
import cv2
import numpy as np
from io import BytesIO
from PIL import Image       
# Create your views here.
def home(request, *args, **kwargs ):
    if request.method == "POST":
        file = request.FILES['file']
        result = predict(file)
        return JsonResponse({"result": result})
    return render(request, 'home.html')

def predict(file):
    model = load_model('app/model.h5')
    img = Image.open(BytesIO(file.read()))
    # img = img.resize((200, 200))
    # img = img.convert('L')  # Convert image to grayscale
    # img = np.array(img)
    # img = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img, axis=3)
    # img = img / 255.0  # Normalize the image

    gray_img = img.convert('L')
    resized_img = gray_img.resize((200, 200))
    img_array = np.array(resized_img)
    edges = cv2.Canny(img_array, 100, 200)
    edge_map = np.reshape(edges, (edges.shape[0], edges.shape[1], -1))
    imgToPred = np.expand_dims(edge_map, axis=0)
    
    # Make predictions
    result = model.predict(imgToPred)
    predicted_class = np.argmax(result)
    labels = ['Glioma Tumor','Meningioma Tumor','No Tumor','Pituitary Tumor']
    print("predicted class: ",predicted_class)
    print("result: ",result)
    return labels[predicted_class]
