from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import predict_class
import os

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save to a temporary location
        fs = FileSystemStorage(location='temp/')  # Specify the temp directory
        filename = fs.save(image_file.name, image_file)  # Save the file
        img_path = fs.url(filename)  # Get the file URL for rendering in HTML

        # Use the predict_class function to get the prediction result and description
        predicted_class, description = predict_class(os.path.join('temp', filename))  # Use the full path

        # Pass both the predicted class, description, and uploaded image URL to the result.html page
        return render(request, 'detection/result.html', {
            'predicted_class': predicted_class,
            'description': description,
            'uploaded_file_url': img_path  # Pass the image URL to display the image in result.html
        })

    return render(request, 'detection/upload.html')
