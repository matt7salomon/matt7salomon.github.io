
<head>
<link rel="shortcut icon" type="image/x-icon" href="favicon.ico?">
</head>
# About
I am a seasoned data scientist that specializes in prototyping analytical solutions at scale on big data. Currently, my day to day work goes with writing python, sql, spark code, leading teams, leading projects, and developing strategy. I have both IC and team lead experience. I have experience with:

`LLMs • Fine-tuning • Generative AI • Transformers • Machine learning • time series analysis • predictive analytics • deep learning • Natural language processing • Vector and Graph databases • dashboards and visualizations • Teaching data science • Connecting with non-technical audience`

I have experience developing the following solutions:

`Health insurance risk • Recommendation Engines • A/B testing • Image processing • Real-time object detection • Fraud Detection • Ad/Promotion Targeting • Developing LLM chatbots • Fine tuning LLMs • RAG • Customer Relation Management (CRM) • Enterprise Resource Planning • Business Insights • Voice assistants and Chatbots`

I currently actively use the following tools depending on client needs:

`Python • R • Spark • SQL • Scala • SAS • Java • Tableau • Shiny App • Hive • Hadoop • Linux • MySQL •Neo4j`

Notable packages and libraries that I use for data mining and machine learning:

`Pyspark • Spark ML • MLlib • Scikit-learn • Theano • TensorFlow • Pytorch • H2O • Transformers`

I have used the following cloud platforms and services:

`AWS • Azure • GCP • Databricks • Domino • S3 • Redshift • TDV • BigQuery • Snowflake`

![image](https://github.com/user-attachments/assets/5a2f54f7-6720-4d46-acc1-92289096d244)

# Notable Corporate and Consulting Projects:

### Legal Navigator (GPT4 chatbot):
This was a 0 to 1 product ideation to deployment into production. Created a GPT4 based RAG using the langchain and azure ML ecosystem that would connect and vectorize thousands of files in different languages and respond to the customer's legal questions using those files. The chatbot would include the filename and the paragraphs where it got its answer from. The backend was using a FastAPI which was deployed in production in Posit-connect and the front-end was using chainlit and streamlit. The backend was using an Azure Search AI vector store and performing an innovative hybrid search which filtered the irrelevant vectors quickly for a faster response time. 

### Fine-tuning GPT 3.5 turbo and GPT 4o mini for feature extraction:
Fine-tuned GPT 3.5 turbo and GPT 4o mini to receive complaint files submitted to a law firm as input and extract the necessary features that lawyers spent hundreds of hours to extract in the past. The submitted complaint files went through an OCR and text cleaning process. The fine-tuning was done on a scale.ai and Azure-ML platform and the dataset for fine-tuning was extracted using an API call to a relativity database that the lawyers had previously used. The training dataset included 3100 complaints which were all less than 10 pages but the LLM was tuned to be able to get larger files in the validation phase as well. Extracted features were not all straightforward and the LLM had to draw conclusions from text to be able to extract some of the features. The accuracy was evaluated by Mistral AI LLM and was between 80% to 98% for different fields but the majority of fields had an accuracy of more than 95%.

### Fine-tuning Llama2 model for question answering based on previous customer calls:
This was a 0 to 1 product ideation to deployment into production. Fine-tuned a llama2 model on a databricks platform to be able to respond to questions that the account executives had on the products from the saved chat between system's engineers and the customers that were saved as scripts. The product reduced the need for account executives to reach out to technical personnel to get information about the products within the company portfolio.

### Health Insurance Risk:
Evaluated risk of small corporate accounts with under 250 members to decide if the insurance company should extend a quote or adjust a quote for these customers. Our model decided increments, decrements, or decline to quote for small corporate customers where historical claims data did not exist for a 2 year ongoing engagement. The provider was previously using a prescription only model and I helped them extended to a prescription+medical model improving metrics and saving the company money.
> [!NOTE]
> My Med+RX model is currently in production for this large health insurance provider. It is being retrained periodically on the newly provided data to adjust to medication price inflations and to avoid model drift but the structure stays the same.

### Health Claims Acceptance/denial:
A major part of this project was feature engineering to extract meaningful features from the available data to be able to predict if a certain claim should be accepted or denied. Our team of 4 data scientists split efforts on different models and feature engineering to be able to reach an accuracy, precision, and recall of all around 95%. Our champion model was an AdaBoost classifier which even performed better than the neural networks which tended to overfit on this particular dataset. The whole process which took on average of 4 days to respond by the healthcare provider was simplified to minutes in this case.

### Prompt Engineering:
Worked on prompt engineering of GPT and llama based models to achieve the desired results in terms of output format and content. The prompt engineering included zero-shot and few-shot learning and was applied to non fine-tuned models.

### LLM Performance Evaluation and Hallucination Reduction:
Used Mistral AI to evaluate the performance of GPT 3.5 responses matching the features in the golden dataset. Also, used the newly developed Ragas module in python to get the context relatedness, context precision, context recall, accuracy,... for the extracted features by an LLM. Also, used newly developed approaches in the literature to reduce hallucination of openai LLMs beyond just adjusting temperature.

### LLM Safety and Conversational AI:
I created some conversational AI chatbots that were able to speak to the customer and respond back with the answers. To keep customer data safe, I used offline fine-tuned LLMs for this purpose. I have some toy examples of the code in my github repo. One of these examples uses a moduel GPT4all that allows to easily download and fine-tune opensource LLMs and the other one is using the huggingface transformers module. While most of my codes ran in the cloud, my github codes are tuned to use the GPUs on my Mac M1. I had also used different strategies in fine-tuning to keep the fine-tuning data safe from being exposed to malicious attacks.

### Search Engine A/B Test and product A/B test: 
Analyzed two search engine products by calculating total utilization and remuneration from Ad clicks per day. The test was performed by dividing traffic equally to two search engines and determining which one was more successful in terms of ad-clicks. Also, on the same project, I performed an A/B test between some minimal changes to the website figuring out which version performed better in terms of attracting viewers to click on different buttons.

### Sales Forecasting:
Wrote an elaborate code to perform sales forecasting using ARIMA and Facebook Prophet for the future quarter for many clients. The code considered many possible elements such as seasonality or hosted events that would have an impact on sales. Also, wrote a win-model which was a classifier predicting which sales opportunities have a higher chance to be won so the account executive spend more time on the more profitable opportunities.

### ERP and CRM using Neo4j Graph database:
This project was performing some tasks with planning and management data loaded into Neo4j. The dataset had been loaded using some Cypher queries. I am not a master at Cypher but can make it work. The rest of the project was parsing the data and performing machine learning on the parsed dataset.

### Recommendation Engine (Similar to Netflix recommendations): 
This was a 0 to 1 product ideation to deployment into production. Built a recommendation engine in Spark to recommend new movies to users. The algorithm was placed in a Microsoft scheduler and executed twice a day. It would recommend new movies to the users in different carousels based on their viewer-ship history, recent purchases, and recent clicks. The product included a recency element where the most recent user ratings would get more weight. It also included a cold-start algorithm that was working based on user-segmentation from the features the customer had collected from users. 
> [!NOTE]
> This product is still in production in a major telecommunication company on their hadoop cluster. The recommendation engine has been upgraded a few times but it is still essentially the same base product. It is being used to recommend videos in south american countries that the telecommunication company serves.

### Detecting cancer from mammograms (deep learning): 
This project was a deep learning use-case to detect cancer from mammograms and had to have a high recall. The final results would be triaged by a physician and it was important that all positive cases are caught. The results were presented to the customer using metrics such as accuracy, precision, and recall.

### Real time object detection in the road for a self driving car (deep learning): 
Used deep learning and a YOLO algorithm to detect objects from thousands of images and videos collected by cars dashcam. Per clients request the method was implemented in Tensorflow, Caffe, Pytorch, and mxnet for a quick comparison. The final deployed algorithm would detect pedestrians, trucks, SUVs, sedans, and bicycles as well as road signs.
> [!NOTE]
> This product went into production for a Japanese car manufacturer. Not sure if they still use this version or changed to a different product as YOLO has advanced and improved a lot since.

### Natural Language Processing (NLP): 
This project included many parts: one was sentiment analysis of servicenow tickets to find out which product was more successful in deployment. Another part was topic modeling of a bank's previous pdfs on environmental and social issues. The other was using a Bert model to find masked words within the documents. We also used the same distil-Bert model from HuggingFace to answer questions from the large textual data corpus of the bank as GPT 3 model had not been released yet.

### Creating and Training a GPT2 model from Scratch (self-experiment):
I created a GPT2 model with embedding, positional encoding, multi-head self attention, feed-forward layer, and normalization. I trained the model distributed with internet data using a cloud provider since it was too large to fit on a laptop. GPT2 is very similar to GPT3 in structure with less parameter and the structure source code is available in the transformers module from huggingface. The results are shared in a repo on my github.

### Fine-tuning Llama3 on a mac silicone M1 (self-experiment):
I was one of the first people to fine-tune a Llama3 8billion parameter on a Mac device and share the code on github. This code connects to huggingface and downloads the model and finetunes it on a public dataset. The reason this was challenging is that Macs dont have a Cuda GPU and bitsandbytes python module still doesnt support Mac so I had to find workarounds online.

### Marketing Channel Attribution (finding the most successful ad platforms): 
wrote an object oriented code to share credit between offline an online marketing channels for user sign-ups using collected data from user’s browsing history. The client had used Facebook, google, TV, radio,... ads and wanted to find out which ads were most successful to readjust their marketing strategy. 

### Converting legacy code between python/R/Java/Spark/Scala:
On many occasions, I had to convert legacy code between different platforms for many reasons including adapting it to a cloud provider or to connect to a new platform or to make it work as a distributed processing code. 

### Ad/promotion targeting:
Helped a software vendor target ads and promotions to the customers who were likely to churn not to disturb all customers at the same time and to save money on promotions too. The solution would send a pop up to on the screen of the targeted user to sign-up again at a discount.

### Fraud Analysis and Anomaly Analysis: 
Detected fraud accounts in millions of customer accounts based on their pattern compared to similar accounts in the same domain as well as other unusual patterns

### Lost and Found Images for a Courier (deep learning): 
We received a large dataset of objects found by the courier in their lost and found with item id and the item picture from different angels. Customers had also submitted pictures of their lost items. We trained a convolutional neural network to match the customer submitted photo to the pictures in the repository and find the lost item. In this instance, the client had approached us with a well defined problem but we created the solution and deployed it into production for the courier. 

### Subscription Roadmap: 
Found the most successful paths to user subscriptions from customer road including clicks, website visits, store logins, etc.

### Customer Lifetime Value (CLV) and Churn: 
Predicted churn along with the CLV of the customer base by clustering similar customers using all collected features.  I also estimated the potential value from any subscription using deep learning.

### Stock/Options Portfolio Risk: 
This project was to optimize a portfolio of stocks and options for different clients based on their age, risk tolerance, liquidity, wealth and some other factors.

### Cybersecurity Vulnerability Detection: 
This project was to find common vulnerabilities within the firewall system based on the data collected by the server log. We identified common points of attacks and weaknesses and vulnerabilities using a data driven approach. This project did not involve penetration testing or other forms of test that are not data driven.

### Root Cause Analysis: 
Detected and isolated the root-cause of failure in large petrochemical processes from collected sensor data (pressure, temperature, flow, etc.) using advanced data-mining methods.
> [!NOTE]
> We submitted three patent applications on this product and all the three were approved.

### Multivariate solar forecasting:
Used some time series sensor reading such as ApparentTemperature, AirTemperature, DewPointTemperature, RelativeHumidity, WindSpeed, WindDirection to forecast the solar generation using an RNN network. Experimented with very different LSTM based RNN strcutures for best results. 
> [!NOTE]
> This project was submitted to California Public Utilities Commision.

### RAG with Llama3 and GPT4o mini:
Created a RAG process that used chainlit as the frontend and Llama3 or GPT4o mini as the LLM in the backend. This app could load many docx, PDF, and txt files and vectorize them in a chromadb vector database. It was then able to answer to user questions in chainlit.

![image](https://github.com/user-attachments/assets/9c86e9bf-81e0-4f27-8a2f-891776569109)

       
# Teaching Data Science:
Taught the following courses in University of California at long beach, and San Diego State University: <br/>
- Data Storytelling <br/>
- Big Data <br/>
- Advanced SQL <br/>
- Intro to Data Science <br/>
- Python Programming <br/>
- Statistics and Probability

# Github Repo Projects:
### LLM and NLP Projects:
- A Rag implemented with Chainlit and llama3. This RAG is free with all open-source packages such as chromadb vector store and works nicely: <br/>
https://github.com/matt7salomon/RAG_chainlit_llama3 <br/> <br/>
- I was one of the first people to fine-tune Llama3 on my mac m1 device on a public dataset: <br/>
https://github.com/matt7salomon/finetuning-llama3-on-macM1 <br/> <br/>
- This is a very simple bias testing of LLMs which only includes a few observations but showing the capability: <br/>
https://github.com/matt7salomon/LLM_bias_testing <br/> <br/>
- LLM long summarization using Langchain and Llama3.1 or GPT 3.5 as backend. This repo also includes a code for translation of the summary as well: <br/> 
https://github.com/matt7salomon/LLM_long_text_summarization_translation <br/> <br/>
- This repo mainly uses the Kor package in python in conjunction with mistralAI or GPT or other LLMs to extract structured data from text and perform an evaluation on the results as well. <br/>
https://github.com/matt7salomon/mistralAI_feature_extraction_LLM_performance_analysis <br/> <br/>
- This code uses the newly available RAGAS module to do a prformance analysis on RAG results. It shows metrics like Faithfulness, Answer relevancy, Context recall, Context precision, Context utilization, Context entity recall, Noise Sensitivity, Summarization Score. It also measures the harmfulness of the LLM responses. <br/>
https://github.com/matt7salomon/ragas_RAG_LLM_performance_analysis <br/><br/>
- This is another implementation of RAG which uses Chainlit as frontend and OpenAI LLMs as required (such as GPT4) for backend. <br/>
https://github.com/matt7salomon/RAG_chainlit_openai <br/><br/>
- This is my implementation of GPT2 from scratch with multihead attention, feedforward NN, and positional encoding functions: <br/>
https://github.com/matt7salomon/GPT2_from_scratch<br/><br/>
- This my sample customer support chatbot which uses a process to finetune llama3 for the purpose. Currently its using a tiny question and answer dataset for fine-tuning and its to show capability: <br/>
https://github.com/matt7salomon/finetuning-llama3-customersupport-chatbot <br/><br/>
- This is an NLP finetuning project for the uncased-bert model on a Macbook M1 and uses GPU. This is a great example for adapting a small model for sentence classification. <br/>
https://github.com/matt7salomon/BERT_Fine_Tuning_Sentence_Classification_Macbook_M1 <br/><br/>
- A document classification code which is simple and understandable. This is really an NLP project: <br/>
https://github.com/matt7salomon/document_classification <br/><br/>
- This is a search engine that uses TF-IDF. Its not the best algorithm for real world applications but does the job in this case. This is really another NLP project: <br/>
https://github.com/matt7salomon/search_engine <br/><br/>
- This shows a brief experiment with Gensim python package for NLU and NLP. The examples are simple and understandable <br/>
https://github.com/matt7salomon/gensim_NLP_examples <br/><br/>
- This is an AI voice assistant that works on a Mac M1 with an offline LLM which comes from GPT4all. There are several LLMs to choose from and several whisper models from Openai to choose from. Also, the wakeup command can be adjusted to the desired command. <br/>
https://github.com/matt7salomon/voiceAI_voiceassistance_gpt4all <br/><br/>
- This code is for a conversational AI using different offline models on a Macbook M1 and it works perfectly even with simple LLMs like GPT2. <br/>
https://github.com/matt7salomon/conversationalAI-Macbook-M1-GPT2  <br/><br/>

### Deep learning projects:
- A sample classifier with tensorflow to show code structure mainly: <br/> 
https://github.com/matt7salomon/my_sample_tensorflow_classifier_fromScratch <br/> <br/>
- This is my project for the IBM data scientist certificate and it usese a 1 dimensional CNN and Xgboost for the implementation. It also has a lot of EDA and exploration and metrics for evaluation as this was presented to the class: <br/>
https://github.com/matt7salomon/xgboost_and_cnn_sample <br/><br/>
- I added my CNN deep learning code in tensorflow which outperforms resnet50 and resnet50 finetuned. Detecting cancer from these images is really challenging. <br/>
https://github.com/matt7salomon/breast_cancer_from_mammograms_deep_learning <br/><br/>
- This is an implementation of yolo7 directly from the relevant paper that I adjusted to work on my mac m1 GPU. It works great and a vide from my tesla dashcam has been annotated by it already. <br/>
<li class="masthead__menu-item"> <a href= https://github.com/matt7salomon/yolo7-mac-M1-tesla-dashcam >Research</a> </li>  <br/><br/>
here is the annotated video: https://drive.google.com/file/d/1oaZsGFmIWwoYR2lW_zbZgi2qvsoM7L04/view?usp=drive_link
### Time series forecasting: 
- This is my great implementation of RNN time series prediction which includes a rolling window for the data to be fed into the RNN and acheives great results. <br/>
https://github.com/matt7salomon/RNN_timeseries_prediction_rollingwindow <br/><br/>
- A very old implementation of Solar forecasting using RNNs. The results are relatively good though. This is a multivariate prediction and includes exogenous inputs. <br/>
https://github.com/matt7salomon/solar_forecasting <br/><br/>
- Another multivarite RNN timeseries prediction with exogenous inputs. <br/>
https://github.com/matt7salomon/multivariate_RNN_timeseries_exogenous <br/><br/>
- This is the time series course by Jose Portilla which I find very helpfull. <br/>
https://github.com/matt7salomon/timeseries_course <br/><br/>
- My hypothesis testing experiment on Uber public data. <br/>
https://github.com/matt7salomon/hypothesis_testing <br/><br/>
- Writing an RNN for time series prediction from scratch: <br/>
https://github.com/matt7salomon/RNN_timeseries_prediction_fromscratch <br/><br/>
- This is an implementation of time series forecasting using DARTS python package which includes RNNs, Prophet, ARIMA, NBEATS, and other models.<br/>
https://github.com/matt7salomon/forecast_timeseries_darts_macbook_m1 <br/><br/>
- This is my time series anomaly detection model. For now its using an isolation forest but will be adding more models here. <br/>
https://github.com/matt7salomon/timeseries-anomaly-detection <br/><br/>
### Graph Networks and Graph Neural Network (GNN):
- This project is using a neo4j on a macbook m1 desktop installation for Enterprise Resource Planning (ERP). Although this implementation is simple, it shows capability. <br/>
https://github.com/matt7salomon/Enterprise_Resource_Planning_Graph_Database <br/><br/>
- This project is using a neo4j on a macbook m1 desktop installation for Customer Relationship Management (CRM). It includes simple and understandable steps to connect to the graph database and create or modify customers, products, and purchases. This project also includes a notebook with a simple machine learning applied to a data that had been loaded to Neo4j in a previous cell using a cypher query. The data for machine learning is a synthetic CRM data. <br/>
https://github.com/matt7salomon/Customer_Relationship_Management_Neo4J_Graph_Database/tree/main <br/><br/>
- This code is an implementation of GNN on Cora citations dataset which is a general benchmark. The GNN was implemented in Pytorch and currently only with Cpu support. I will add Mac and Cuda GPU support later. <br/>
https://github.com/matt7salomon/GNN_Pytorch_Cora_Citation_Data/tree/main <br/><br/>
### Transformers module:
- This a bert model that i used for sentiment analysis on a public dataset. I did a similar project on servicenow tickets as well: <br/>
https://github.com/matt7salomon/bert_sentiment_analysis <br/><br/>
- This is my implementation of transformers from scratch to show capability: <br/>
https://github.com/matt7salomon/transformers_from_scratch <br/><br/>
### Reinforcement learning:
- This is a sample reinforcement learning code that uses the Openai Gym package to solve some games by a trial/error rewards method. Simple code and understandable.  <br/>
https://github.com/matt7salomon/game-solving-with-reinforcement-learning-Openai-gym  <br/><br/>
### Other projects:
- This is a sample EDA that I had to perform for a certificate: <br/>
https://github.com/matt7salomon/sample_EDA <br/><br/>
- This is my anomaly detection code which includes many methods but I prefer histograms, z-score, and isolation forests for this purpose. <br/>
https://github.com/matt7salomon/anomaly_detection <br/><br/>
- I developed a FastAPI module and wrapped in a pip installable package to show capability. <br/>
https://github.com/matt7salomon/pipinstallable <br/><br/>


