import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import os
import gdown

# Load your model
# model = load_model("detection/models/model.h5")  # Update the path to your model

# Define the Google Drive file ID and the destination path
file_id = "1GcoUqB6WhT0BeqD4nbqVJy4yJfO61v4Z"
model_path = "detection/models/model.h5"

# Check if the model file exists locally; if not, download it
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load your model
model = load_model(model_path)

def predict_class(img_path):
    img = Image.open(img_path)
    img = img.resize((299, 299))  # Resize to match model input shape
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range

    # Make a prediction using your model
    prediction = model.predict(img_array)

    # Assuming you have a mapping of class indices to class names
    class_indices = {
        0: 'bbps-0-1',
        1: 'bbps-2-3',
        2: 'cecum',
        3: 'dyed-lifted-polyps',
        4: 'dyed-resection-margins',
        5: 'esophagitis-a',
        6: 'polyps',
        7: 'pylorus',
        8: 'retroflex-rectum',
        9: 'retroflex-stomach',
        10: 'ulcerative-colitis-grade-1-2',
        11: 'z-line'
    }

    class_descriptions = {
        0: ("bbps-0-1: This category represents early-stage lesions that are typically benign. "
            "These lesions are often detected during routine screenings and are generally not associated "
            "with significant health risks. However, monitoring is essential to ensure they do not evolve "
            "into more serious conditions. Early detection allows for minimal intervention, which is often "
            "more effective and less invasive than treatment for advanced lesions."),
            
        1: ("bbps-2-3: These lesions are of intermediate complexity and may require closer monitoring. "
            "They can indicate potential progression of underlying issues and necessitate further diagnostic "
            "evaluation to rule out malignancy. In some cases, these lesions might show atypical features, "
            "suggesting a need for more comprehensive imaging or biopsy to assess their nature and determine "
            "the appropriate management strategies."),
        
        2: ("cecum: This region is part of the large intestine and may exhibit various abnormalities, "
            "including tumors, inflammation, and other pathological changes. The cecum is a critical area for "
            "screening as it can harbor significant lesions that may lead to serious complications if left untreated. "
            "Imaging studies focused on the cecum can reveal important diagnostic information, facilitating timely intervention."),
        
        3: ("dyed-lifted-polyps: These polyps have been enhanced with dye for better visualization during "
            "endoscopic procedures. The dye helps delineate the edges and morphology of polyps, allowing for more "
            "accurate assessment of their characteristics. Proper identification and management of these polyps are crucial, "
            "as certain types can be precursors to colorectal cancer."),
        
        4: ("dyed-resection-margins: This indicates margins after polyp removal, assessed for cancerous cells. "
            "Ensuring clear margins is vital to minimize the risk of recurrence. The examination of these margins is "
            "a crucial step in the post-operative evaluation process to confirm that all potentially malignant tissues "
            "have been successfully excised."),
        
        5: ("esophagitis-a: This indicates inflammation of the esophagus, which may result from reflux or infections. "
            "Symptoms can include difficulty swallowing, chest pain, and heartburn. Management often includes lifestyle "
            "modifications and medication to reduce acid reflux and promote healing of the esophageal lining."),
        
        6: ("polyps: These are abnormal tissue growths that can be benign or precancerous. Polyps may vary in size "
            "and type, and their presence in the colon or rectum can indicate an increased risk of colorectal cancer. "
            "Regular screening and removal of polyps are essential to prevent cancer development."),
        
        7: ("pylorus: This is the region of the stomach that connects to the small intestine. Conditions affecting the pylorus, "
            "such as pyloric stenosis or cancer, can lead to significant gastrointestinal issues. Monitoring and assessment "
            "of this area are important for diagnosing and managing potential complications."),
        
        8: ("retroflex-rectum: This refers to backward bending of the rectum, often seen in diagnostic imaging. "
            "This condition can impact the ability to visualize and assess the rectal lining for abnormalities. "
            "Thorough evaluation of the retroflexed area is essential to detect any lesions or pathological changes."),
        
        9: ("retroflex-stomach: Similar to the rectum, this indicates backward bending of the stomach. "
            "This positioning can complicate the visualization of gastric lesions during endoscopy. Careful examination is "
            "needed to ensure that any lesions in this area are accurately identified and managed."),
        
        10: ("ulcerative-colitis-grade-1-2: This indicates mild to moderate ulcerative colitis, a chronic inflammatory bowel disease. "
            "Patients may experience symptoms such as abdominal pain, diarrhea, and rectal bleeding. Management typically involves "
            "anti-inflammatory medications and regular monitoring to prevent progression of the disease."),
        
        11: ("z-line: This is the junction between the esophagus and stomach, crucial for reflux assessment. "
            "Changes in the z-line can indicate gastroesophageal reflux disease (GERD) or Barrett's esophagus. "
            "Evaluation of the z-line is important for diagnosing esophageal conditions and planning appropriate treatment.")
    }

    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_indices[predicted_class_index]
    description = class_descriptions[predicted_class_index]

    return predicted_class, description
