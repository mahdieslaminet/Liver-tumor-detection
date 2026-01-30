
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# --- 1. Define Custom Metrics (REQUIRED to load the model) ---
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# --- 2. Load the Model ---
# This assumes the model file is in the same folder as this script
model = tf.keras.models.load_model('model_best.keras', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})

# --- 3. Prediction Function ---
def predict_segmentation(image):
    # Resize to 256x256
    img_resized = cv2.resize(image, (256, 256))
    
    # Grayscale conversion
    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_resized
        
    # Normalize and Reshape (Model expects: Batch, Height, Width, Channel)
    img_input = img_gray.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Predict
    pred_mask = model.predict(img_input)
    output_mask = pred_mask[0, :, :, 0]
    
    # Threshold (0.5) to make it binary
    binary_mask = (output_mask > 0.5).astype(np.uint8) * 255
    return binary_mask

# --- 4. Launch Interface ---
interface = gr.Interface(
    fn=predict_segmentation,
    inputs=gr.Image(label="Upload CT Slice"),
    outputs=gr.Image(label="AI Prediction"),
    title="Liver Tumor Segmentation",
    description="Upload a CT scan slice to see the segmented liver area."
)

interface.launch()
