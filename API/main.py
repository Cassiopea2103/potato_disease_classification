from fastapi import FastAPI , File , UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1/potato_disease_prediction.keras")
CLASS_NAMES = ['Early Blight' , 'Late Blight' , 'Healthy']

# function to read file as image :
def read_file_as_image ( data ) -> np.ndarray :
    image = np.array ( Image.open ( BytesIO ( data ) ) )
    return image



@app.post ('/predict')
async def predict (
        file: UploadFile = File (...)
) :
    image = read_file_as_image ( await file.read () )
    # transforming the image to a batch , since the model only takes batches:
    img_batch = np.expand_dims(image , 0)

    # making predictions on the image :
    predictions = MODEL.predict ( img_batch )

    # the model returns % for each category predictions .
    # the highest % should be the best prediction :
    predicted_class = CLASS_NAMES [ np.argmax(predictions[0])]
    confidence = np.max ( predictions[0])


    return {
        'class' : predicted_class ,
        'confidence' : float ( confidence ) 
    }





if __name__ == "__main__":
    uvicorn.run ( app , port = 2103 , host = 'localhost' , )