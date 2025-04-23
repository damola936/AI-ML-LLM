import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import tensorflow as tf
import pandas as pd

# 🔹 Expand the Page Layout
st.set_page_config(layout="wide")  # Forces full-width mode

current_model = "Model Mini"

class_names = ['apple_pie',
               'baby_back_ribs',
               'baklava',
               'beef_carpaccio',
               'beef_tartare',
               'beet_salad',
               'beignets',
               'bibimbap',
               'bread_pudding',
               'breakfast_burrito',
               'bruschetta',
               'caesar_salad',
               'cannoli',
               'caprese_salad',
               'carrot_cake',
               'ceviche',
               'cheese_plate',
               'cheesecake',
               'chicken_curry',
               'chicken_quesadilla',
               'chicken_wings',
               'chocolate_cake',
               'chocolate_mousse',
               'churros',
               'clam_chowder',
               'club_sandwich',
               'crab_cakes',
               'creme_brulee',
               'croque_madame',
               'cup_cakes',
               'deviled_eggs',
               'donuts',
               'dumplings',
               'edamame',
               'eggs_benedict',
               'escargots',
               'falafel',
               'filet_mignon',
               'fish_and_chips',
               'foie_gras',
               'french_fries',
               'french_onion_soup',
               'french_toast',
               'fried_calamari',
               'fried_rice',
               'frozen_yogurt',
               'garlic_bread',
               'gnocchi',
               'greek_salad',
               'grilled_cheese_sandwich',
               'grilled_salmon',
               'guacamole',
               'gyoza',
               'hamburger',
               'hot_and_sour_soup',
               'hot_dog',
               'huevos_rancheros',
               'hummus',
               'ice_cream',
               'lasagna',
               'lobster_bisque',
               'lobster_roll_sandwich',
               'macaroni_and_cheese',
               'macarons',
               'miso_soup',
               'mussels',
               'nachos',
               'omelette',
               'onion_rings',
               'oysters',
               'pad_thai',
               'paella',
               'pancakes',
               'panna_cotta',
               'peking_duck',
               'pho',
               'pizza',
               'pork_chop',
               'poutine',
               'prime_rib',
               'pulled_pork_sandwich',
               'ramen',
               'ravioli',
               'red_velvet_cake',
               'risotto',
               'samosa',
               'sashimi',
               'scallops',
               'seaweed_salad',
               'shrimp_and_grits',
               'spaghetti_bolognese',
               'spaghetti_carbonara',
               'spring_rolls',
               'steak',
               'strawberry_shortcake',
               'sushi',
               'tacos',
               'takoyaki',
               'tiramisu',
               'tuna_tartare',
               'waffles']

top_ten_dict = {
    "class_name": ["edamame", "macarons", "oysters", "pho",
                   "mussles", "sashimi", "seaweed_salad", "dumplings", "guacamole", "onion_rings"],
    "f1-score": [0.964427, 0.900433, 0.853119, 0.852652, 0.850622,
                 0.844794, 0.834356, 0.833006, 0.83209, 0.831967]
}

last_ten_dict = {
    "class_name": ["chocolate_mousse", "tuna_tartare",
                   "scallops", "huevos_rancheros", "foie_gras", "steak",
                   "bread_pudding", "ravioli", "pork_chop", "apple_pie"],
    "f1-score": [0.413793, 0.399254, 0.383693, 0.367698,
                 0.354497, 0.340426, 0.340045, 0.339785, 0.324826, 0.282407]
}

# 🔹 Custom CSS for Full Width & Centered Content
st.markdown(
    """
    <style>
        /* Make the main container wider */
        .main-container {
            max-width: 95% !important;
            margin: auto;
        }

        /* Center all content inside containers */
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            width: 100%;
        }

        .centeredh {
            display: flex;
            width: 80%;
        }

        /* Ensure file uploader is not constrained */
        div[data-testid="stFileUploader"] {
            width: 70% !important;
        }

        /* Center images */
        img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px;
            height: 200px;
            border-radius: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Food vision demo App 🍔🧠")
st.header(
    "A food vision app, using a Machine Learning Model(CNN), fine tuned on EfficientNet.")

st.divider()
st.subheader("What is a CNN(Convolutional Neural Network)")
st.write("A Neural network is network of nodes, consiting of input nodes, output nodes and hidden nodes.\
    Each node lies in its respective layer, corresponding to its name. \
         The input nodes reside in the input layer, the output nodes reside in the output layer and the hidden\
            nodes reside in the hidden layer. The nodes pass information from the input layer to the output layer.\
                The information consists of data(text, numbers, pictures, audio, videos) encoded as numbers\
                    that the network uses to learn information. It does this through complex mathematical operations\
                        and algorithms.")

# Display image of Neural Network here in between dividers

st.write("A Convolutional Neural Network in short is a version\
     of a Neural Network that specializes on Images, video, basically anything visual.")

st.divider()
code = """import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable mixed precision
mixed_precision.set_global_policy("mixed_float16")

image_shape = (224, 224, 3)

# Load EfficientNet with mixed precision
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=image_shape, name="input_layer")

# Apply data augmentation
x = data_augmentation(inputs)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

x = tf.keras.layers.Dense(len(train_data.class_names), name="dense_logits")(x)

# Ensure output layer remains in FP32
outputs = tf.keras.layers.Activation(activation="softmax", dtype=tf.float32, name="predictions")(x)

model = tf.keras.Model(inputs, outputs)

# Use a LossScaleOptimizer to prevent numerical issues
optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=["accuracy"])

# Train the model
history = model.fit(train_data, epochs=5, validation_data=test_data,
                    validation_steps=int(0.15 * len(test_data)),
                    callbacks=[create_tensorboard_callback("model_mini", "model"), checkpoint_callback])"""
st.subheader("Sample Code for the CNN using TensorFlow Functional API using Transfer Learning (NOT FULL CODE)")
st.code(code, language="python")
st.divider()

st.divider()

st.subheader("What is Efficient Net")
st.write("EfficientNet is a family of convolutional neural networks that are designed to be more efficient and accurate. \
    It scales up the model's width, depth, and resolution in a balanced way, which helps to achieve better performance \
    with fewer resources. In simple terms, EfficientNet can achieve high accuracy on image classification tasks while \
    using less computational power and memory compared to other models.")

st.divider()
st.subheader("What is Fine Tuning")
st.write("Fine-tuning is a process in machine learning where a pre-trained model is further trained on a new, but related, dataset. \
    This helps the model to adapt to the new data and improve its performance on specific tasks. \
    Essentially, it takes advantage of the knowledge the model has already gained and refines it for better accuracy.")

st.divider()
tune_code = """# Load feature extraction weights
model.load_weights(checkpoint_path)

# Unfreeze all layers in the base model
base_model.trainable = True

# Freeze all layers except the last 5
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Use a LossScaleOptimizer to prevent numerical issues
optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())

# Recompile the Model with Lower Learning Rate to reduce overfitting
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=optimizer, metrics=["accuracy"]) # Learning rate lowered by 10x

model_tuned_history = model.fit(train_data, epochs=10, initial_epoch=history.epoch[-1],
                                validation_data=test_data, validation_steps=int(0.15 * len(test_data)),
                                callbacks=[create_tensorboard_callback("model_mini", "model_tuned")])"""

st.subheader("Example of Fine Tuning Using TensorFlow (NOT FULL CODE)")
st.code(tune_code, language="python")

st.divider()
st.subheader("Model Building Details")
st.write(f'The Model was built using the :blue[Food101 kaggle dataset].\
 The Dataset consist of 101 classes of Food.\
     Namely: {[food.replace("_", "").title() for food in class_names]}')

st.divider()
st.write("When predicting you have to pass an image of any of the 101 classes of food.\
     The Model has not yet been trained outside the 101 classes of food yet.")

st.divider()
st.subheader("Top and Least Classes Performance.")
st.write("After training, some classes evidently performed better than others.\
     Below are the performance of the top classes and least classes based on the F1 score")

st.divider()
st.subheader("F1-score")
st.write("The F1 score is a measure of a test's accuracy, which considers both the precision and the recall of the test to compute the score. The F1 score is the harmonic mean of precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. \
    Precision is the number of true positive results divided by the number of all positive results, including those not correctly identified (i.e., the proportion of positive identifications that were actually correct). \
        Recall (or Sensitivity) is the number of true positive results divided by the number of positives that should have been identified (i.e., the proportion of actual positives that were correctly identified).")

st.divider()
st.subheader("The formula for F1-score is")
st.latex(r"F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
st.divider()

# Top 10 last 10 Bar charts
st.subheader("Top and Least Classes")
with st.container():
    st.markdown('<div class="centeredh">', unsafe_allow_html=True)  # START DIV BLOCK

    top_ten = pd.DataFrame(top_ten_dict).sort_values("f1-score", ascending=False)
    last_ten = pd.DataFrame(last_ten_dict).sort_values("f1-score", ascending=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top 10 Classes.")
        st.bar_chart(top_ten, x="class_name", y="f1-score",
                     horizontal=True, use_container_width=True)

    with col2:
        st.write("Last 10 classes")
        st.bar_chart(last_ten, x="class_name", y="f1-score",
                     horizontal=True, use_container_width=True, color="#ff748c")

    st.markdown('</div>', unsafe_allow_html=True)  # CLOSE DIV BLOCK

new_model = "Food Vision"
st.divider()
st.divider()
st.header(f"Try out the Current Models, :blue[{current_model}] and :blue[{new_model}] your self.")
st.caption("_The Model is periodically being improved. Model might change in the future_.")


def load_model(filepath):
    """
    Loads a Tensorflow keras Model from a file path

        Args:
            filepath(str): File path to the Model.

        Returns
            A Tensorflow keras loaded Model.
    """
    with st.spinner("Loading Model..."):
        try:
            loaded_model = tf.keras.models.load_model(filepath)
        except Exception as e:
            st.error(f"Can't load Model: {e}")
        else:
            if loaded_model:
                return loaded_model


def load_prep_image(image: UploadedFile, img_shape=224, scale=True):
    """
        Reads in an image and preprocesses it for model prediction

        Args:
            image (UploadedFile): path to target image
            img_shape (int): shape to resize image to. Default = 224
            scale (bool): Condition to scale image. Default = True

        Returns:
            Image Tensor of shape (img_shape, img_shape, 3)
    """
    bytes_data = image.getvalue()
    image_tensor = tf.io.decode_image(bytes_data, channels=3)
    image_tensor = tf.image.resize(image_tensor, [img_shape, img_shape])
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Expand dimension as needed by Model
    if scale:
        scaled_image_tensor = image_tensor / 255.  # If model does not have built in scaling
        return scaled_image_tensor
    else:
        return image_tensor


def predict_using_model(image: UploadedFile, model_path: str) -> str:
    """
        This function uses the CNN Model to predict the class name of the uploaded
        input image.

        Args:
            model_path(str): The path to the Model
            image(UploadedFile Object): the uploaded image.
        
        Returns:
            predicted_class_name(str): the name of the predicted class.
    """
    with st.spinner("Predicting using your image..."):
        # Process the image
        processed_image = load_prep_image(image, scale=False)  # EfficientNet has built in scaling
        model = load_model(model_path)
        pred_prob = model.predict(processed_image)
        predicted_class = class_names[pred_prob.argmax()]  # Get the predicted class name

    return predicted_class


def toggle_checkbox(option: str) -> None:
    """Toggle upload checkboxes such that only one can be selected"""
    if option == "upload":
        st.session_state.upload = True
        st.session_state.camera = False
    elif option == "camera":
        st.session_state.upload = False
        st.session_state.camera = True


def toggle_model(option: str) -> None:
    """Toggles model checkboxes such that only one can be selected"""
    if option == "model_mini":
        st.session_state.model_mini = True
        st.session_state.food_vision = False
    elif option == "food_vision":
        st.session_state.model_mini = False
        st.session_state.food_vision = True


# 🔹 Apply the main container styling
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 🔹 Create a wider main container
with st.container():
    # Define columns inside the main container
    cols = st.columns([3, 1, 2, 1, 2], gap="medium")
    has_predicted = False
    has_uploaded = False

    # 🖼️ Image Input Container
    with cols[0]:
        with st.container():
            st.markdown('<div class="centered">', unsafe_allow_html=True)  # START DIV BLOCK

            with st.spinner("Uploading image..."):
                try:
                    upload = st.checkbox("Upload Image", key="upload",
                                         on_change=toggle_checkbox, args=("upload",))
                    camera = st.checkbox("Use your camera", key="camera",
                                         on_change=toggle_checkbox, args=("camera",))
                    if upload:
                        uploaded_image = st.file_uploader(label="Upload an image (Max 200MB)",
                                                          type=["png", "jpg", "jpeg"],
                                                          accept_multiple_files=False, key="uploaded_image")

                        has_uploaded = True  # To check if file_uploader widget has loaded

                        if "uploaded_image" not in st.session_state:
                            st.session_state["uploaded_image"] = uploaded_image

                    elif camera:
                        uploaded_image = st.camera_input("Take a Picture",
                                                         disabled=not camera, key="uploaded_image")

                        has_uploaded = True  # To check if camera_input widget has loaded

                        if "uploaded_image" not in st.session_state:
                            st.session_state["uploaded_image"] = uploaded_image

                except Exception as e:
                    st.error(f"Image Upload failed: {e}")
                else:
                    if has_uploaded:  # If file_uploader/camera_input widget has loaded
                        if uploaded_image:  # If user has uploaded an image
                            st.success("Image Uploaded.")
                            st.image(st.session_state.uploaded_image,
                                      caption="Your uploaded image", width=200)

            st.markdown('</div>', unsafe_allow_html=True)  # CLOSE DIV BLOCK

    # ➡️ Arrow 1 Container
    with cols[1]:
        with st.container():
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            st.write("➡️")  # Example arrow to be changed to image
            st.markdown('</div>', unsafe_allow_html=True)

    # 🧠 Neural Network Image Container
    with cols[2]:
        with st.container():
            st.markdown('<div class="centered">', unsafe_allow_html=True)

            st.write("Pick a Model")
            model_mini = st.checkbox("Model Mini", key="model_mini",
                                      on_change=toggle_model, args=("model_mini",))
            food_vision = st.checkbox("Food Vision", key="food_vision",
                                       on_change=toggle_model, args=("food_vision",))

            if model_mini:
                st.image("brain.png")
            elif food_vision:
                st.image("content/creativity_15557951.png")  # To be changed

            if has_uploaded:
                status = st.button(label="Predict Using Image", icon="⚛️", type="primary")
                if status and model_mini:
                    result_class = predict_using_model(uploaded_image,
                                                        model_path="models/model_mini_Food101.keras")
                    has_predicted = True
                elif status and food_vision:
                    result_class = predict_using_model(uploaded_image, model_path="models/FoodVision.keras")
                    has_predicted = True

            st.markdown('</div>', unsafe_allow_html=True)

    # ➡️ Arrow 2 Container
    with cols[3]:
        with st.container():
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            st.write("➡️")  # Example arrow to be changed to image
            st.markdown('</div>', unsafe_allow_html=True)

    # 🏆 Output Container
    with cols[4]:
        with st.container():
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            if has_predicted:
                st.image(st.session_state.uploaded_image)
                if "_" in result_class:
                    modified_class = result_class.replace("_", "").title()
                    st.write(f"This is an image of :blue[{modified_class}]")
                else:
                    st.write(f"This is an image of :blue[{result_class.title()}]")
            else:
                st.write("The Image and Prediction will appear here")
            st.markdown('</div>', unsafe_allow_html=True)

# Close the widened container
st.markdown('</div>', unsafe_allow_html=True)
