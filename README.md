# Student_score_prediction
This project analyzes and predicts student exam performance based on various factors such as study hours, attendance, and other lifestyle attributes. It applies data analysis, visualizations, and machine learning models (Linear & Polynomial Regression) to understand patterns and improve prediction accuracy.
Dataset
The dataset contains student-related attributes that may influence academic performance. Example features include:
Hours_Studied – Number of study hours per week
Attendance – Student’s attendance percentage
Sleep_Hours – Average sleep hours per day
Exam_Score – Final exam score (target variable)

Project Workflow
Importing Libraries – pandas, numpy, matplotlib, seaborn, scikit-learn
Loading Dataset – Reading Student_Performance.csv into a DataFrame
Exploratory Data Analysis (EDA)
Distribution plots
Correlation heatmap
Scatter plots (Study Hours vs Exam Score)
Model Training
Linear Regression (straight-line relationship)
Polynomial Regression (captures nonlinear trends)
Evaluation Metrics
R² Score → Measures accuracy of predictions
MSE (Mean Squared Error) → Lower means better predictions
Visualization – Comparing actual vs predicted exam scores

Machine Learning Models
Linear Regression
Predicts exam score as a linear function of study hours.
Polynomial Regression
Fits a curve to capture more complex relationships.
Multiple Linear Regression (optional)
Uses several features (e.g., attendance, sleep hours) to improve predictions.

Results
Linear Regression provides a baseline prediction.
Polynomial Regression often fits better when data has a nonlinear relationship.
Multiple features (attendance, sleep, etc.) improve accuracy beyond study hours alone.

Author
Developed by [Tehseen ullah saif] 
Feel free to fork this project, raise issues, or suggest improvements!
