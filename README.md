# Project Title: Movie Rating Predictor Application
# Team: 
Emily Bartman, Joseph Shepherd, and Alexander Schmig
# Project Description: 
## Objectives
Our problem is that movie ratings are too retroactive. We will create a dependable movie ratings prediction model to set movie makers’ and audience members' expectations upon the release of a new film, before the critics.
## Usefulness
This application will be useful for two primary reasons: (1) rating expectations impact film financing, and (2) rating expectations impact audience willingness to attend.  While researching some applications showed current ratings like RottenTomatoes, IMDb, and Metacritic, but they do not show predictions of movie ratings. However, there were many articles about utilizing prediction models with no application being created to interact with and allow usage of the models. Below are some of these articles: 

- [New Tomatometer Scores: Latest Ratings on Movies and Shows | Rotten Tomatoes](https://editorial.rottentomatoes.com/article/tomatometer-scores/)
- [Predict Movie Rating | Data Science Blog (nycdatascience.com)](https://nycdatascience.com/blog/student-works/web-scraping/movie-rating-prediction/)
- [Predicting IMDb Movie Ratings using Supervised Machine Learning | by Joe Cowell | Towards Data Science](https://towardsdatascience.com/predicting-imdb-movie-ratings-using-supervised-machine-learning-f3b126ab2ddb)
- [The 4 Recommendation Engines That Can Predict Your Movie Tastes | by James Le | Towards Data Science](https://towardsdatascience.com/the-4-recommendation-engines-that-can-predict-your-movie-tastes-109dc4e10c52)
- [Movie Rating Prediction | Kaggle](https://www.kaggle.com/code/sherinclaudia/movie-rating-prediction)
- [How to Predict Sentiment from Movie Reviews Using Deep Learning (Text Classification) - MachineLearningMastery.com](https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)
- [New AI Predicts Movie Ratings Before Filming | Psychology Today](https://www.psychologytoday.com/us/blog/the-future-brain/202011/new-ai-predicts-movie-ratings-filming)
- [Electronics | Free Full-Text | A Recommendation Engine for Predicting Movie Ratings Using a Big Data Approach (mdpi.com)](https://www.mdpi.com/2079-9292/10/10/1215)
- [26260680.pdf (stanford.edu)](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26260680.pdf)
- [Predicting Movie Ratings with Machine Learning Algorithms | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-51156-2_125)

Based on an initial purview of the market, it appears this is a niche gap of prior research and theory being made available but lacking application, especially in the publicly accessible domain. As such, this is a good opportunity to corner the market and create a publicly accessible page wherein anyone can not only see your theory but perform live interactions with our model and predict future ratings for upcoming films. Our application’s target audience are the groups that benefit the most from a rating prediction tool: filmmakers, cinema owners, streaming service owners, and premier viewers. These stakeholders will find the most benefit out of the tool as they can gain both cost savings and funding based on the predictions gained from our application.

## Dataset
The dataset chosen for our model is public domain licensed from IMDB by web scaping and posted on Kaggle.com 3 years ago by Harhit Sankhdhar. It appears to have been posted for the purposes of public experimentation of the data, but no explicit purpose was posted by Harhit Sankhdhar.

**Name:** IMDB Movies Dataset

**Owner:** HARSHIT SHANKHDHAR

**Sourced:** IMDb 

**Stored:** Kaggle.com 

**Date:** Feb 01, 2021

**URL:** IMDb Dataset of the Top 1000 Movies & TV Shows 

Our initial data summary was performed in Google Colab. The summary shows the data set to have the following characteristics: 

- Shaped: (1000, 16)						 

- Fields and their data type: 

        Poster_Link 		object 
        Series_Title 		object 
        Released_Year 	object 
        Certificate 		object 
        Runtime 		object 
        Genre 	        object 
        IMDB_Rating		float64 
        Overview 	    	object 
        Meta_score 		float64 
        Director 		object 
        Star1 		object 
        Star2 		object 
        Star3 		object 
        Star4 		object 
        No_of_Votes 		int64 
        Gross 		object 

- Stat Description:

                IMDB_Rating    Meta_score    No_of_Votes
        Count   1000.000        843.000        1.000000e+03
        Mean    7.9493          77.97153       2.736929e+05
        std     0.275491        12.376099      3.273727e+05
        min     7.6             28.000         2.508800e+04
        25%     7.7             70             5.552625e+04
        50%     7.9             79             1.385485e+05
        75%     8.1             87             3.741612e+05
        max     9.3             100            2.343110e+06

- NA values: 

      Poster_Link 		0 
      Series_Title 		0 
      Released_Year   	0 
      Certificate 		101
      Runtime 		0
      Genre 			0 
      IMDB_Rating 		0 
      Overview 		0 
      Meta_score 		157 
      Director 		0 
      Star1 			0 
      Star2			0 
      Star3 			0 
      Star4 			0 
      No_of_Votes 		0 
      Gross 			169


Based on these initial findings, we find that cleaning of the data will be necessary because there are missing values, and most field types are object. To clean the data, we plan to primarily utilize the removal of the rows containing the missing data as the dataset is large enough to handle this method.

## Functionalities
Our application will perform a range of data science functions and user interactions. The data science functions include statistical analysis performed on the sample data, visualizations of the data, predictive models used, and training recommendations for continued exploratory analysis with our tool. This will allow us to not only build up our ethos as authorities in this area but also share insight into the Data Science methodologies acquired in this course. 

Furthermore, our application's true value and differentiator in the market is the area for user interactions. The user interactions area of the application will allow the user to enter information about the movie and get a prediction on the rating based on the model chosen. Further user interactions that we may consider exploring are giving the user the ability to train the model on their own dataset and run predictions in parallel for comparison.

To support the ambitions of this project, we will need to further explore the available options for web app technology. However, our team is most experienced with Python, so we plan to focus on efforts around Streamlit and Firebase. We plan to get more hands-on experience in the following weeks on the two technologies before making a final decision. 

## Communication and Sharing 
Our team utilizes Zoom to meet virtually as needed and has set up a WhatsApp group for continuous communications, a Google doc to capture the proposal documentation, a Google Colab page to capture code, and a Github Repo to store our content. Links below.
### Google Doc
Documentation of our proposal:
https://docs.google.com/document/d/1sBhkrY3nsiTtuq2WroSdFMkXmu96rMsqwrlep8_bjcg/edit?usp=sharing 
### Google Colab
Coding collaboration site: 
https://colab.research.google.com/drive/12tohs0uisfzyGIftprB9KUNOH7K32a7x?usp=sharing 
### Github Repo
Our GitHub repo contains a Read.me file with this initial project description saved with the dataset. To check it out, visit the below link:
https://github.com/EmilyBartman/ADS_Group4_Final_Project_Repo

# Personal Contribution Statement

**Team Member:** Emily Bartman

- **Contribution:** My contribution during the planning stage (Part 1) was setting up and participating in our WhatsApp group and Zoom meetings, creating the GitHub Repo, drafting the Final Project Proposal doc, uploading the read.me file, and reviewing the submission. 

**Team Member:** Joseph Shepherd

- **Contribution:** My contribution during the planning stage (Part 1) was primarily a reviewer of the document, as well as uploading the data to the shared GitHub repository.  I also shared ideas and strategies during the team meetings and via our WhatsApp group.

**Team Member:** Alexander Schmig

- **Contribution:** My contribution during the planning stage (Part 1) was finding the dataset on Kaggle, performing statistical analysis of the dataset, participating in meetings and WhatsApp communications, and reviewing the doc for submission.



# References

Awan, M. J., Khan, R. A., Nobanee, H., Yasin, A., Anwar, S. M., Naseem, U., & Singh, V. P. (2021). A Recommendation Engine for Predicting Movie Ratings Using a Big Data Approach. Electronics, 10(10), 1215. https://doi.org/10.3390/electronics10101215

Brownlee, J. (2016, July 3). How to Predict Sentiment From Movie Reviews Using Deep Learning (Text Classification). Machine Learning Mastery. https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

Çağlıyor, S., & Başar Öztayşi. (2020). Predicting Movie Ratings with Machine Learning Algorithms. Advances in Intelligent Systems and Computing, 1077–1083. https://doi.org/10.1007/978-3-030-51156-2_125

CLAUDIA, S. (2019, July 2). Movie Rating Prediction. Kaggle.com. https://www.kaggle.com/code/sherinclaudia/movie-rating-prediction

Cowell, J. (2020, October 15). Predicting IMDb Movie Ratings Using Supervised Machine Learning. Medium. https://towardsdatascience.com/predicting-imdb-movie-ratings-using-supervised-machine-learning-f3b126ab2ddb

Le, J. (2018, June 11). The 4 Recommendation Engines That Can Predict Your Movie Tastes. Medium. https://towardsdatascience.com/the-4-recommendation-engines-that-can-predict-your-movie-tastes-109dc4e10c52

Rosso, C. (2020, October 18). New AI Predicts Movie Ratings Before Filming | Psychology Today. Www.psychologytoday.com. https://www.psychologytoday.com/us/blog/the-future-brain/202011/new-ai-predicts-movie-ratings-filming

SHANKHDHAR, H. (2021, February 1). IMDB Movies Dataset. Www.kaggle.com. https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

Staff, R. (2023, October 13). New Tomatometer Scores: Latest Ratings on Movies and Shows. Https://Editorial.rottentomatoes.com/. 
https://editorial.rottentomatoes.com/article/tomatometer-scores/

Sun, C. (2016, August 22). Predict Movie Rating | Data Science Blog. Nycdatascience.com. https://nycdatascience.com/blog/student-works/web-scraping/movie-rating-prediction/

Yang, Y., Ma, R., & Cho, M. (2019). Predicting Movie Ratings with Multimodal Data. https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26260680.pdf
