from flask import Flask
from CleanData import Clean_bp
from AIAlg.Batch.Fully_Connected_Neural_Networks import Customer_Classification_F_bp, Market_Segmentation_F_bp, Sales_Prediction_bp
from AIAlg.Batch.CNN_ import Image_Analysis_bp, Feature_Extraction_bp
from AIAlg.Batch.RNN_ import Predict_Sales_bp
from AIAlg.Batch.LSTM_ import Analyze_Customer_Behavior_bp
from AIAlg.Batch.Collaborative_Filtering import Recommendations_bp, Suggest_products_bp
from AIAlg.Batch.Content_Based_Filtering import Recommendations_By_Features_bp, Recommendations_By_User_History_bp
from AIAlg.Batch.Autoencoder import Autoencoder_Anomaly_bp
from CleanData.Batch import Autoencoder_Reduction_bp
from AIAlg.Batch.Decision_Tree import Customer_Classification_D_bp, Market_Segmentation_D_bp, Rule_Based_Decision_bp
from AIAlg.Batch.Random_Forest import Customer_Behavior_bp, Customer_Classification_R_bp, Sales_Estimation_bp
from AIAlg.Batch.Gradient_Boosting_Machines import Classification_bp, Prediction_Improvement_bp, Regression_bp
from AIAlg.Batch.Support_Vector_Machines import Svm_Classification_bp, Svm_Fraud_Detection
from AIAlg.Batch.ARIMA import Arima_Sales_Forecast_bp, Arima_Seasonal_Analysis_bp
from AIAlg.Batch.Prophet_Facebook import Prophet_Forecast_bp

app = Flask(__name__)


#PULIZIA DATI
app.register_blueprint(Clean_bp)  # Pulizia dati grezzi con pyspark
#Batch
#Autoencoder
app.register_blueprint(Autoencoder_Reduction_bp)

#BATCH
#Fully Connected Neural Networks
app.register_blueprint(Sales_Prediction_bp)
app.register_blueprint(Customer_Classification_F_bp)
app.register_blueprint(Market_Segmentation_F_bp)
#CNN
app.register_blueprint(Image_Analysis_bp)
app.register_blueprint(Feature_Extraction_bp)
#RNN
app.register_blueprint(Predict_Sales_bp)
#LSTM
app.register_blueprint(Analyze_Customer_Behavior_bp)
#Collaborative Filtering (Filtraggio Collaborativo)
app.register_blueprint(Recommendations_bp)
app.register_blueprint(Suggest_products_bp)
#Content-Based Filtering (Filtraggio Basato sul Contenuto)
app.register_blueprint(Recommendations_By_Features_bp)
app.register_blueprint(Recommendations_By_User_History_bp)
#Autoencoder
app.register_blueprint(Autoencoder_Anomaly_bp)
#Decision Tree
app.register_blueprint(Customer_Classification_D_bp)
app.register_blueprint(Market_Segmentation_D_bp)
app.register_blueprint(Rule_Based_Decision_bp)
#Random Forest
app.register_blueprint(Customer_Behavior_bp)
app.register_blueprint(Customer_Classification_R_bp)
app.register_blueprint(Sales_Estimation_bp)
#Gradient Boosting Machines
app.register_blueprint(Classification_bp)
app.register_blueprint(Prediction_Improvement_bp)
app.register_blueprint(Regression_bp)
#Support Vector Machines
app.register_blueprint(Svm_Classification_bp)
app.register_blueprint(Svm_Fraud_Detection)
#ARIMA
app.register_blueprint(Arima_Seasonal_Analysis_bp)
app.register_blueprint(Arima_Sales_Forecast_bp)
#Prophet Facebook
app.register_blueprint(Prophet_Forecast_bp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
