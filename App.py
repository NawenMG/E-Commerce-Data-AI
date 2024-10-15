from flask import Flask
from CleanData import Clean_bp
from AIAlg.Batch.Fully_Connected_Neural_Networks.Customer_Classification_F_bp import customer_classification_bp
from AIAlg.Batch.Fully_Connected_Neural_Networks.Market_Segmentation_F_bp import market_segmentation_bp
from AIAlg.Batch.Fully_Connected_Neural_Networks.Sales_Prediction_bp import sales_prediction_bp
from AIAlg.Batch.CNN_ import Feature_Extraction_bp
from AIAlg.Batch.CNN_ import Image_Analysis_bp
from AIAlg.Batch.RNN_.Predict_Sales_bp import predicate_Sales_RNNLSTM_bp
from AIAlg.Batch.LSTM_.Analyze_Customer_Behavior_bp import analyze_customer_behavior_bp
from AIAlg.Batch.Collaborative_Filtering.Recommendations_bp import recommendations_bp
from AIAlg.Batch.Collaborative_Filtering.Suggest_products_bp import suggest_products_bp
from AIAlg.Batch.Content_Based_Filtering.Recommendations_By_Features_bp import recommendations_by_features_bp
from AIAlg.Batch.Content_Based_Filtering.Recommendations_By_User_History_bp import recommendations_by_user_history_bp
from AIAlg.Batch.Autoencoder.Autoencoder_Anomaly_bp import autoencoder_anomaly_bp
from CleanData.Batch.Autoencoder_Reduction_bp import autoencoder_reduction_bp
from AIAlg.Batch.Decision_Tree.Customer_Classification_D_bp import decision_tree_customer_classification_bp
from AIAlg.Batch.Decision_Tree.Market_Segmentation_D_bp import decision_tree_market_segmentation_bp
from AIAlg.Batch.Decision_Tree.Rule_Based_Decision_bp import decision_tree_rule_based_decision_bp
from AIAlg.Batch.Random_Forest.Customer_Behavior_bp import random_forest_customer_behavior_bp
from AIAlg.Batch.Random_Forest.Customer_Classification_R_bp import random_forest_customer_classification_bp
from AIAlg.Batch.Random_Forest.Sales_Estimation_bp import random_forest_sales_estimation_bp
from AIAlg.Batch.Gradient_Boosting_Machines.Classification_bp import gradient_boosting_classification_bp
from AIAlg.Batch.Gradient_Boosting_Machines.Prediction_Improvement_bp import gradient_boosting_prediction_improvement_bp
from AIAlg.Batch.Gradient_Boosting_Machines.Regression_bp import gradient_boosting_regression_bp
from AIAlg.Batch.Support_Vector_Machines.Svm_Classification_bp import svm_classification_bp
from AIAlg.Batch.Support_Vector_Machines.Svm_Fraud_Detection_bp import svm_fraud_detection_bp
from AIAlg.Batch.ARIMA.Arima_Sales_Forecast_bp import arima_sales_forecast_bp
from AIAlg.Batch.ARIMA.Arima_Seasonal_Analysis_bp import arima_seasonal_analysis_bp
from AIAlg.Batch.Prophet_Facebook.Prophet_Forecast_bp import prophet_forecast_bp
from AIAlg.Streaming.LSTMRNN.lstm_sales_prediction import lstm_sales_bp;
from AIAlg.Streaming.LSTMRNN.Gru_anomaly import gru_anomaly_bp
from AIAlg.Streaming.CNN_1D.cnn_ClickStream import cnn_clickstream_bp
from AIAlg.Streaming.Transformers.transformer_recommendation import transformer_recommendation_bp
from AIAlg.Streaming.Modelli_Anomaly_Detection.autoencoder_fraud_detection import autoencoder_fraud_bp
from AIAlg.Streaming.Modelli_Anomaly_Detection.lstm_anomaly_detection import lstm_anomaly_bp
from AIAlg.Streaming.RL.reinforcement_learning_recommendation import rl_recommendation_bp
from AIAlg.Streaming.Online_Learning.online_learning import online_learning_bp
from AIAlg.Streaming.TFX.sales_forecasting import forecasting_bp


app = Flask(__name__)

#PULIZIA DATI
app.register_blueprint(Clean_bp, url_prefix='/clean_data')  # Pulizia dati grezzi con pyspark
#Batch
#Autoencoder
app.register_blueprint(autoencoder_reduction_bp, url_prefix='/reduce_dimensionality')


#BATCH
#Fully Connected Neural Networks
app.register_blueprint(sales_prediction_bp, url_prefix='/predict_sales')
app.register_blueprint(sales_prediction_bp, url_prefix='/train_model')
app.register_blueprint(customer_classification_bp, url_prefix='/classify_customers')
app.register_blueprint(customer_classification_bp, url_prefix='/train_model')
app.register_blueprint(market_segmentation_bp, url_prefix='/segment_market')
app.register_blueprint(market_segmentation_bp, url_prefix='/train_model')
#CNN
app.register_blueprint(Image_Analysis_bp, url_prefix='/analyze_image')
app.register_blueprint(Image_Analysis_bp, url_prefix='/train_model')
app.register_blueprint(Feature_Extraction_bp, url_prefix='/extract_features')
app.register_blueprint(Feature_Extraction_bp, url_prefix='/train_model')
#RNNLSTM
app.register_blueprint(predicate_Sales_RNNLSTM_bp, url_prefix='/predict_sales')
app.register_blueprint(predicate_Sales_RNNLSTM_bp, url_prefix='/train_model')
#LSTM
app.register_blueprint(analyze_customer_behavior_bp, url_prefix='/analyze_customer_behavior')
app.register_blueprint(analyze_customer_behavior_bp, url_prefix='/train_model')
#Collaborative Filtering (Filtraggio Collaborativo)
app.register_blueprint(recommendations_bp, url_prefix='/recommendations')
app.register_blueprint(recommendations_bp, url_prefix='/train_model')
app.register_blueprint(suggest_products_bp, url_prefix='/suggest_products')
app.register_blueprint(suggest_products_bp, url_prefix='/train_model')
#Content-Based Filtering (Filtraggio Basato sul Contenuto)
app.register_blueprint(recommendations_by_features_bp, url_prefix='/recommendations_by_features')
app.register_blueprint(recommendations_by_features_bp, url_prefix='/train_model')
app.register_blueprint(recommendations_by_user_history_bp, url_prefix='/recommendations_by_user_history')
app.register_blueprint(recommendations_by_user_history_bp, url_prefix='/train_model')
#Autoencoder
app.register_blueprint(autoencoder_anomaly_bp, url_prefix='/detect_anomalies')
app.register_blueprint(autoencoder_anomaly_bp, url_prefix='/train_model')
#Decision Tree
app.register_blueprint(decision_tree_customer_classification_bp, url_prefix='/classify_customer')
app.register_blueprint(decision_tree_customer_classification_bp, url_prefix='/train_model')
app.register_blueprint(decision_tree_market_segmentation_bp, url_prefix='/segment_market')
app.register_blueprint(decision_tree_market_segmentation_bp, url_prefix='/train_model')
app.register_blueprint(decision_tree_rule_based_decision_bp, url_prefix='/make_decision')
app.register_blueprint(decision_tree_rule_based_decision_bp, url_prefix='/train_model')
#Random Forest
app.register_blueprint(random_forest_sales_estimation_bp, url_prefix='/estimate_sales')
app.register_blueprint(random_forest_sales_estimation_bp, url_prefix='/train_model')
app.register_blueprint(random_forest_customer_classification_bp, url_prefix='/classify_customer')
app.register_blueprint(random_forest_customer_classification_bp, url_prefix='/train_model')
app.register_blueprint(random_forest_customer_behavior_bp, url_prefix='/predict_customer_behavior')
app.register_blueprint(random_forest_customer_behavior_bp, url_prefix='/train_model')
#Gradient Boosting Machines
app.register_blueprint(gradient_boosting_classification_bp, url_prefix='/classify')
app.register_blueprint(gradient_boosting_classification_bp, url_prefix='/train_model')
app.register_blueprint(gradient_boosting_prediction_improvement_bp, url_prefix='/improve_predictions')
app.register_blueprint(gradient_boosting_prediction_improvement_bp, url_prefix='/train_model')
app.register_blueprint(gradient_boosting_regression_bp, url_prefix='/regress')
app.register_blueprint(gradient_boosting_regression_bp, url_prefix='/train_model')
#Support Vector Machines
app.register_blueprint(svm_classification_bp, url_prefix='/classify_customers')
app.register_blueprint(svm_classification_bp, url_prefix='/train_model')
app.register_blueprint(svm_fraud_detection_bp, url_prefix='/detect_fraud')
app.register_blueprint(svm_fraud_detection_bp, url_prefix='/train_model')
#ARIMA
app.register_blueprint(arima_seasonal_analysis_bp, url_prefix='/analyze_seasonality')
app.register_blueprint(arima_seasonal_analysis_bp, url_prefix='/train_model')
app.register_blueprint(arima_sales_forecast_bp, url_prefix='/forecast_sales')
app.register_blueprint(arima_sales_forecast_bp, url_prefix='/train_model')
#Prophet Facebook
app.register_blueprint(prophet_forecast_bp, url_prefix='/forecast')
app.register_blueprint(prophet_forecast_bp, url_prefix='/train_model')


#STREAMING
#LSTM/RNN
app.register_blueprint(lstm_sales_bp, url_prefix='/predict_sales')
app.register_blueprint(lstm_sales_bp, url_prefix='/train_model')
app.register_blueprint(gru_anomaly_bp, url_prefix='/detect_anomalies')
app.register_blueprint(gru_anomaly_bp, url_prefix='/train_model')
#CNN 1D
app.register_blueprint(cnn_clickstream_bp, url_prefix='/predict_clickstream')
app.register_blueprint(cnn_clickstream_bp, url_prefix='/train_model')
#Transformers
app.register_blueprint(transformer_recommendation_bp, url_prefix='/recommend')
app.register_blueprint(transformer_recommendation_bp, url_prefix='/train_model')
#Modelli Anomaly Detection
app.register_blueprint(autoencoder_fraud_bp, url_prefix='/detect_fraud')
app.register_blueprint(autoencoder_fraud_bp, url_prefix='/train_model')
app.register_blueprint(lstm_anomaly_bp, url_prefix='/detect_anomalies')
app.register_blueprint(lstm_anomaly_bp, url_prefix='/train_model')
#Reinforcment learning
app.register_blueprint(rl_recommendation_bp, url_prefix='/recommend')
app.register_blueprint(rl_recommendation_bp, url_prefix='/feedback')
#Online Learning
app.register_blueprint(online_learning_bp, url_prefix='/increment')
app.register_blueprint(online_learning_bp, url_prefix='/predict')
#TFX
app.register_blueprint(forecasting_bp, url_prefix='/forecast')
app.register_blueprint(forecasting_bp, url_prefix='/train_model')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
