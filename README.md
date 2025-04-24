## **Automated Customer Support Ticket Classification Using NLP, Topic Modeling, and Machine Learning.**

#### **Problem Statement**

For a customer-centric company, support tickets are a key feedback mechanism reflecting product/service issues. Traditionally, these unstructured text tickets are manually triaged to appropriate departments, resulting in delayed resolution, human error, and inefficiencies—especially as customer volume increases.

To address this, the project aims to automate ticket classification using Natural Language Processing (NLP) techniques and Machine Learning (ML). By automatically identifying and categorizing the type of issue raised, companies can resolve customer concerns faster and improve operational efficiency.

#### **Project Objective**

To build a robust classification system that:


* Preprocesses and cleans ticket descriptions.

* Identifies key topics using Non-Negative Matrix Factorization (NMF).

* Trains multiple ML classifiers using labeled ticket data.

* Selects the best-performing model for deployment.

* Allows real-time prediction through a Gradio interface.

#### **Dataset**

[https://www.kaggle.com/api/v1/datasets/download/suraj520/customer-support-ticket-datase](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
The dataset used consists of customer support tickets, each with:

* **Ticket Description**: A textual description of the issue.

* **First Response Time**: Timestamp of first agent response.

* **Time to Resolution**: Time taken to resolve the issue.

* **Ticket Type**: Category of the ticket (our target label).

Preprocessing focused on the **Ticket Description**, which required NLP techniques to transform the raw text into useful machine-readable features.

#### **Text Preprocessing**

Text preprocessing involved several stages:

1. **Missing Value Handling**: Dropped records missing key fields (description, response time, resolution time, and type).

2. Timestamp Conversion: Formatted timestamps for further analysis.

3. **Text Cleaning**:

* Lowercased the text.

* Removed punctuation, numbers, placeholders, and repetitive phrases.

* Removed common non-informative phrases (e.g., "please assist", "I bought").

4. **Stopword Removal**: Using NLTK’s stopword list.

5. **Lemmatization**: Applied WordNetLemmatizer to reduce words to their base forms.

6. **Part of Speech (POS) Tagging**: Counted POS tag distributions to understand language patterns.


#### **Exploratory Data Analysis (EDA)**

We performed the following visual analyses:

* **Ticket Type Distribution**: Showed most frequent ticket categories.

* **POS Tag Distribution**: Visualized common grammatical structures (e.g., nouns, verbs).

* **Top Unigrams, Bigrams, Trigrams**: Identified recurring terms and phrases in complaints.

* **Word Cloud**: Displayed frequent keywords from cleaned descriptions.


#### **Feature Extraction**

To convert text into numerical form:

* **TF-IDF Vectorization** was applied, capturing term importance relative to the corpus.


#### **Topic Modeling Using NMF**

* **NMF** was used to extract latent topics from ticket descriptions.

* A pre-determined number of topics was defined (e.g., 10).

* Each complaint was then associated with a dominant topic.

* Top keywords for each topic were examined for interpretability.

This allowed us to enhance the labels used in training, particularly where original labels were vague or missing.


#### **Model Building**

We experimented with several supervised machine learning algorithms:

* **Logistic Regression**

* **Naive Bayes**

* **Decision Tree**

* **Random Forest**

After training and validation, the **Random Forest** model achieved the best performance:


|Metric	               |              Random Forest|
----------------------------------------------------
|Accuracy	           |                      94.3%|
|Precision	           |                      93.7%|
|Recall	               |                      92.9%|
|F1 Score	           |                      93.3%|

The confusion matrix showed high classification accuracy across multiple ticket categories.


#### **Model Deployment Using Gradio**

To allow real-time ticket classification, the final Random Forest model was deployed using **Gradio**.

**Features of the interface:**

* Accepts new ticket descriptions as input.

* Outputs the predicted ticket type.

* Enables non-technical users (e.g., support agents) to use the model easily.

#### **Recommended Process Improvements**

**Improve Login and Authentication Experience**

* Simplify password reset process
* Implement two-factor authentication (2FA) for added security and trust.
* Add clear error messafes and troubleshooting steps for login failures.

**Enhance Software Update Communication**
* Proactively notify users about upcoming updates.
* Provide version compatibility info and rollback options if issues arise.

**Optimize System Performance Monitoring**
* Regularly monitor app performance and set automated alerts for latency spikes or crashes.
* Create a light weight troubleshooting guide for common lag /crash issues.
**Tighten Data Security Measures**
* Educate users on secure account practices.
* Publish transparency reports on resolved security issues.
* Automate alerts for unusal user activity.

**Strengthen Customer Follow-up Systems**
* Implement SLA tracing for ticket responses.
* Use automation(e.g., chatbots or email triggers) to follow up if no agent responds within 12 hours or period agreed upon.
* Prioritize unresolved ticket flags in the helpdesk dashboard.

#### **Conclusion**
This project successfully demonstrates how NLP and machine learning can automate customer ticket classification, leading to faster resolution and improved support quality. The Random Forest model, with its high performance, was deployed for real-world interaction via Gradio.

#### **Next Steps**

* Expand to multi-label classification for tickets with multiple issues.

* Integrate customer satisfaction scores (CSAT) for feedback loops.

* Retrain models periodically to adapt to evolving customer concerns.