from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

model_dir = "/home/oscar/Repos/Pessoal/machine-learning---supervised-classification-learning/saved_model"

tokenizer = AlbertTokenizer.from_pretrained(model_dir)
model = AlbertForSequenceClassification.from_pretrained(model_dir)

class_labels = {
    0: "PRODUCT",
    1: "DATA",
    2: "MARKETING",
    3: "BUSINESS",
    4: "TECHNOLOGY",
    5: "DESIGN",
    6: "LEADERSHIP",
    7: "AI",
}

def predict_course_description(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = class_labels.get(predicted_class, "Unknown category")

    return predicted_label, probabilities

course_description = "LEADERSHIP"
predicted_label, probabilities = predict_course_description(course_description)

print(f"Predicted category: {predicted_label}")
print(f"Probabilities: {probabilities}")
