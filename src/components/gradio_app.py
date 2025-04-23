import gradio as gr
import joblib


#Load the model and vectorizer
model = joblib.load('ticket_classifier_model.pkl')
vectorizer = joblib.load('ticket_vectorizer.pkl')


#Define the label map
target_names= {
    0: 'Software Updates',
    1: 'Performance Issues',
    2: 'Account Access',
    3: 'Unresolved Support',
    4: 'Data Security'
}

# Function to predict a topic for custom text
def ticket_classify(description):
    vectorized_text = vectorizer.transform([description])
    prediction = model.predict(vectorized_text)[0]
    return target_names[prediction]
    
# Create a gardio interface
interface = gr.Interface(
    fn=ticket_classify, #Function for prediction
    inputs = gr.Textbox(lines=3,placeholder="Enter a support ticket description..."), # Input type: text
    outputs = gr.Label(num_top_classes=1), # Output type: text
    title = "Customer Ticket Classification Model",
    description = "Enter a support ticket description to identify its topic."

)

# Launch the app
if __name__=="__main__":
    interface.launch(share=True)