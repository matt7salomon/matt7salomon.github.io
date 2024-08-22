# About
I am a principal data scientist that specializes in prototyping analytical solutions at scale on big data. Currently, my day to day work goes with writing python, sql, and spark code, leading teams, and developing strategy. I have experience with:

LLMs • Fine-tuning • Generative AI • Transformers • Machine learning • time series analysis • predictive analytics • deep learning • Natural language processing • large language models • dashboards and visualizations

I have experience developing the following solutions:

Health insurance risk • Recommendation Engines • A/B testing • Image processing • Real-time object detection • Fraud Detection • Ad/Promotion Targeting • Developing LLM chatbots • Fine tuning LLMs • RAG • Business Insights

I currently actively use the following tools depending on client needs:

Python • R • Spark • SQL • Scala • SAS • Java • Tableau • Shiny App • Hive • Hadoop • Linux • MySQL

Notable packages and libraries that I use for data mining and machine learning:

Pyspark • Spark ML • MLlib • Scikit-learn • Theano • TensorFlow • Pytorch • H2O • Transformers

I have used the following cloud platforms and services:
AWS • Azure • GCP • Databricks • Domino • S3 • Redshift • TDV • BigQuery • Snowflake

# Recent Projects

### Legal Navigator (GPT4 chatbot):
Created a GPT4 based RAG using the langchain and azure ML ecosystem that would connect and vectorize thousands of files in different languages and respond to the customer's legal questions using those files. The chatbot would include the filename and the paragraphs where it got its answer from. The backend was using a FastAPI which was deployed in production in Posit-connect and the front-end was using chainlit and streamlit. The backend was using an Azure Search AI vector store and performing a innovative hybrid search which filters the irrelevant vectors quickly for a faster response time.

### Fine-tuning GPT 3.5 turbo and GPT 4o mini for feature extraction:
Fine-tuned GPT 3.5 turbo and GPT 4o mini to receive complaint files submitted to a law firm as input and extract the necessary features that lawyers spent hundreds of hours to extract in the past. The fine-tuning was done on a scale.ai and Azure-ML platform and the dataset for fine-tuning was extracted using an API call to relativity database that the lawyers had previously used.

### Fine-tuning Llama2 model for question answering based on previous customer calls:
Fine-tuned a llama2 model on a databricks platform to be able to respond to questions that the account executives had on the products from the saved chat between system's engineers and the customers that were saved as scripts. The product reduced the need for account executives to reach out to technical personnel to get information about the products within the company portfolio.

### Health Insurance Risk:
Evaluated risk of small corporate accounts with under 250 members to decide if the insurance company should extend a quote or adjust a quote for these customers. Our model decided increments, decrements, or decline to quote for small corporate customers where historical claims data did not exist for a 2 year ongoing engagement. The provider was previously using a prescription only model and I helped them extended to a prescription+medical model improving metrics and saving the company money.

### Search Engine A/B Test: 
Analyzed two search engine products by calculating total utilization and remuneration from Ad clicks per day. The test was performed by dividing traffic equally to two search engines and determining which one was more successful in terms of ad-clicks.

### Sales Forecasting:
Wrote an elaborate code to perform sales forecasting using ARIMA and Facebook Prophet for the future quarter for many clients. The code considered many possible elements such as seasonality or hosted events that would have an impact on sales. Also, wrote a win-model which was a classifier predicting which sales oppurtunities have a higher chance to be won so the account executive spend more time on the more profitable opportunities.

### Recommendation Engine (Similar to Netflix recommendations): 
Built a recommendation engine in Spark to recommend new movies to users. The algorithm was placed in a Microsoft scheduler and executed twice a day. It would recommend new movies to the users in different carousels based on their viewer-ship history, recent purchases, and recent clicks. The product included a recency element where the most recent user ratings would get more weight. It also included a cold-start algorithm that was working based on user-segmentation from the features the customer had collected from users. 

### Detecting cancer from mammograms (deep learning): 
This project was a deep learning use-case to detect cancer from mammograms and had to have a high recall. The final results would be triaged by a physician and it was important that all positive cases are caught. The results were presented to the customer using metrics such as accuracy, precision, and recall.

### Real time object detection in the road for a self driving car (deep learning): 
Used deep learning and a YOLO algorithm to detect objects from thousands of images and videos collected by cars dashcam. Per clients request the method was implemented in Tensorflow, Caffe, Pytorch, and mxnet for a quick comparison. The final deployed algorithm would detect pedestrians, trucks, SUVs, sedans, and bicycles as well as road signs.

### Natural Language Processing (NLP): 
This project included many pasts: one was sentiment analysis of servicenow tickets to find out which product was more successful in deployment. Another part was topic modeling of a bank's previous pdfs on environmental and social issues. The other was using a Bert model to find masked words within the documents.

### Marketing Channel Attribution (finding the most successful ads): 
wrote an object oriented code to share credit between offline an online marketing channels for user sign-ups using collected data from user’s browsing history. The client had used Facebook, google, TV, radio,... ads and wanted to find out which ads were most successful to readjust thier marketing strategy. 

### Converting legacy code between python/R/Java/Spark/Scala:
On many occasions, I had to convert legacy code between different platforms for many reasons including adapting it to a cloud provider or to connect to a new platform or to make it work as a distributed processing code. 

### Ad/promotion targeting:
Helped a software vendor target ads and promotions to the customers who were likely to churn not to disturb all customers at the same time and to save money on promotions too. The solution would send a pop up to on the screen of the targeted user to sign-up again at a discount.

### Fraud Analysis and Anomaly Analysis: 
Detected fraud accounts in millions of customer accounts based on their pattern compared to similar accounts in the same domain as well as other unusual patterns

### Subscription Roadmap: 
Found the most successful paths to user subscriptions from customer road including clicks, website visits, store logins, etc.

### Customer Lifetime Value (CLV) and Churn: 
Predicted churn along with the CLV of the customer base by clustering similar customers using all collected features.  I also estimated the potential value from any subscription using deep learning.

### Root Cause Analysis: 
Detected and isolated the root-cause of failure in large petrochemical processes from collected sensor data (pressure, temperature, flow, etc.) using advanced data-mining methods.

