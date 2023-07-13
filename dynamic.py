import tkinter as tk
from tkinter import filedialog # allows file selection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Loading CSV file
def load_csv():
    root = tk.Tk()
    root.withdraw() # prevents tkinter window from opening
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    data = pd.read_csv(file_path)
    print(data)
    return data 

# Selecting the variables
def select_variables(data):
    print("Header names: ", data.columns)
    independent_vars= input("Enter the independent variable header names: ").split(",")
    dependent_var= input("Enter the dependent variable header names: ") 
    return independent_vars, dependent_var

'''# Linear Regression
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model)
    return model'''

# Training
def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),  
        'Lasso Regression': Lasso(alpha=1.0),  
        'Decision Tree Regression': DecisionTreeRegressor(max_depth=2),
        'Support Vector Regression': SVR()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(model)
    return trained_models

# Prediction
def predict(model, X):
    predictions = model.predict(X)
    #print(predictions)
    return predictions
    
# Main
def main():
    data = load_csv()
    rows = len(data.axes[0]) # no. of rows
    columns = len(data.axes[1]) # no. of column
    print("No. of rows:", rows, "  No. of columns:", columns)
    if rows > 100: print("Enough data")
    else: print("Not enough data", exit()) 
    independent_vars , dependent_var = select_variables(data)
    X = data[independent_vars]
    y = data[dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    '''model = train_model(X_train, y_train) 
    train_predictions = predict(model, X_train)
    test_predictions = predict(model, X_test)'''
    
    trained_models = train_models(X_train, y_train)

    for name, model in trained_models.items():
        train_predictions = predict(model, X_train)
        test_predictions = predict(model, X_test)

        # R-squared score
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        print(f"{name} Training R-squared:", train_r2)
        print(f"{name} Testing R-squared:", test_r2)

        # Root Mean Squared Error (RMSE)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        print(f"{name} Training RMSE:", train_rmse)
        print(f"{name} Testing RMSE:", test_rmse)

        # Mean Absolute Error (MAE)
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        print(f"{name} Training MAE:", train_mae)
        print(f"{name} Testing MAE:", test_mae)

        # Mean Squared Error (MSE)
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        print(f"{name} Training MSE:", train_mse)
        print(f"{name} Testing MSE:", test_mae)
    
if __name__ == "__main__":
    main()
    
        
    '''# R-squared score
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    print("Training R2_score:", train_r2)
    print("Testing R2_score:", test_r2)
    
    # Root Mean Squared Error (RMSE)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print("Training RMSE:", train_rmse)
    print("Testing RMSE:", test_rmse)

    # Mean Absolute Error (MAE)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    print("Training MAE:", train_mae)
    print("Testing MAE:", test_mae)

    # Mean Squared Error (MSE)
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    print("Training MSE:", train_mse)
    print("Testing MSE:", test_mse)'''

    '''# Plot between true and predicted values
    plt.scatter(y_train, train_predictions, label='Training Data')
    plt.scatter(y_test, test_predictions, label='Testing Data') 
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True Values vs Predicted Values')
    plt.legend()
    plt.show()'''

    '''# Accuracy score
    train_accuracy = accuracy_score(y_train, train_predictions.round())
    test_accuracy = accuracy_score(y_test, test_predictions.round())
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

    # ROC-AUC curve
    train_roc_auc = roc_auc_score(y_train, train_predictions)
    test_roc_auc = roc_auc_score(y_test, test_predictions)
    print("Training ROC-AUC:", train_roc_auc)
    print("Testing ROC-AUC:", test_roc_auc)

    fpr_train, tpr_train, _ = roc_curve(y_train, train_predictions)
    fpr_test, tpr_test, _ = roc_curve(y_test, test_predictions)

    plt.figure()
    plt.plot(fpr_train, tpr_train, label='Train ROC curve')
    plt.plot(fpr_test, tpr_test, label='Test ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()'''



