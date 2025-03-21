import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Chargement du modèle CNN sauvegardé
model = tf.keras.models.load_model("bestmodel.h5")
IMG_SIZE = (
    224,
    224,
)  # Doit correspondre à la taille utilisée lors de l'entraînement
SEUIL_OPTIMAL = 0.64
# Dictionnaire de correspondance entre les indices et les classes
labels = {0: "Non Conforme", 1: "Conforme"}


def main():
    st.title("Contrôle Qualité : Vis")
    st.write(
        "OkVisFactory est une application permettant diagnostiquer la qualité d'une vis."
    )

    uploaded_file = st.file_uploader(
        "Chargez une image de vis (format .jpg/.png)...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Affichage de l'image chargée
        image = Image.open(uploaded_file)
        st.image(image, caption="Vis chargée", use_container_width=True)

        # Prétraitement de l'image pour le modèle :
        # 1. Redimensionnement
        image_resized = image.resize(IMG_SIZE)
        # 2. Conversion en tableau numpy
        image_array = np.array(image_resized)
        # Si l'image possède un canal alpha (transparence), le supprimer
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]
        # 3. Normalisation des pixels dans [0, 1]
        image_array = image_array / 255.0
        # 4. Ajout d'une dimension pour le batch
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction avec le modèle
        predictions = model.predict(image_array)
        diagnostic = (
            labels[0] if predictions[0][0] < SEUIL_OPTIMAL else labels[1]
        )

        st.write(f"**Diagnostic :** {diagnostic}")

        if diagnostic == "Conforme":
            st.success("Vis de bonne qualité !")
        else:
            st.error("Défaut détecté !")


if __name__ == "__main__":
    main()
