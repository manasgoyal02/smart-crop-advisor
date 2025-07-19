import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import google.generativeai as genai # type: ignore
import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def plot_crop_probabilities(input_features):
    probabilities = rf.predict_proba(input_features)[0]
    crop_names = [reverse_crop_dict[i] for i in range(1, len(probabilities)+1)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(crop_names, probabilities, color='skyblue')
    ax.set_xticklabels(crop_names, rotation=90)
    ax.set_ylabel("Probability")
    ax.set_title("Crop Prediction Probabilities")
    fig.tight_layout()

    st.pyplot(fig)


crop = pd.read_csv("Crop_Recommendation.csv")

crop_dict = {
    'Rice': 1, 'Maize': 2, 'ChickPea': 3, 'KidneyBeans': 4, 'PigeonPeas': 5,
    'MothBeans': 6, 'MungBean': 7, 'Blackgram': 8, 'Lentil': 9, 'Pomegranate': 10,
    'Banana': 11, 'Mango': 12, 'Grapes': 13, 'Watermelon': 14, 'Muskmelon': 15,
    'Apple': 16, 'Orange': 17, 'Papaya': 18, 'Coconut': 19, 'Cotton': 20,
    'Jute': 21, 'Coffee': 22
}

crop['crop_num'] = crop['Crop'].map(crop_dict)

X = crop[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y = crop['crop_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", acc_rf)

reverse_crop_dict = {v: k for k, v in crop_dict.items()}

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = rf.predict(features)
    print("Predicted class:", prediction[0])
    if prediction[0] in reverse_crop_dict:
        print("The best crop to grow is:", reverse_crop_dict[prediction[0]])
    else:
        print("Invalid prediction")
    return prediction[0]

def get_valid_ph():
    while True:
        try:
            ph = float(input("Enter pH (between 3.5 and 9.0): "))
            if 3.5 <= ph <= 9.0:
                return ph
            else:
                print("Invalid pH. Please enter a value between 3.5 and 9.0.")
        except ValueError:
            print("Please enter a valid numeric value for pH.")




def main():
    st.title("🌾 Crop Recommendation System")

    N = st.number_input("Nitrogen", value=50.0)
    P = st.number_input("Phosphorus", value=30.0)
    K = st.number_input("Potassium", value=40.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=70.0)
    ph = st.slider("pH Value", min_value=3.5, max_value=9.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=100.0)

    if st.button("🌱 Recommend Crop"):
        predicted_crop_num = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        predicted_crop = reverse_crop_dict[predicted_crop_num]
        st.success(f"✅ The best crop to grow is: **{predicted_crop}**")

        plot_crop_probabilities(np.array([[N, P, K, temperature, humidity, ph, rainfall]]))

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            f"The recommended crop is {predicted_crop}. What are the next steps a farmer should follow to grow this crop successfully?"
        )
        st.subheader("💡 Expert Advice from Gemini")
        st.write(response.text)



if __name__ == "__main__":
    main()


