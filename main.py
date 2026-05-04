# main.py

from fake_news_detector import train_model

# Train the model and get the vectorizer
model, vectorizer = train_model()

# Command-line prediction loop
while True:
    print("\n📰 Enter a news headline (or type 'exit' to stop):")
    user_input = input(">> ")
    
    if user_input.lower() == "exit":
        print("👋 Exiting.")
        break

    # Preprocess input
    input_text = user_input.lower()

    # Transform input
    input_tfidf = vectorizer.transform([input_text])

    # Predict
    prediction = model.predict(input_tfidf)

    # Output result
    if prediction[0] == 'REAL':
        print("✅ Prediction: REAL news")
    else:
        print("❌ Prediction: FAKE news")