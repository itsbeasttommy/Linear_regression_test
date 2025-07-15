# Linear_regression_test
Below is a detailed report of how we handled the test as a group

Section 1: Exploratory Data Analysis
Q1: Load the dataset smartphones.csv using pandas
Explanation:
To load the dataset into the notebook, we used the pandas library. The dataset was read using the pd.read_csv() function, which allows us to load CSV files into a pandas DataFrame, making it easy to analyze the data.
Q2: Display the shape of the dataset
Explanation:
We used the shape attribute of the DataFrame (df.shape) to display the dimensions of the dataset. This provided the number of rows and columns, helping us understand the dataset's structure. The dataset has 980 rows and 26 columns.
Q3: Show all unique values in the Brand column
Explanation:
To identify the unique brands in the dataset, we used the unique() method on the "brand_name" column of the DataFrame. This returned an array containing all distinct values for the brand names across all rows, providing an overview of the different smartphone brands present.
Q4: Check for missing values in the dataset
Explanation:
We checked for missing data using the isnull().sum() method, which calculates the number of missing (NaN) values in each column. This helps to identify which attributes in the dataset have incomplete information that might require handling, such as imputation or removal.
Q5: Find the average RAM of all smartphones
Explanation:
To find the average RAM capacity of the smartphones, we computed the mean of the "ram_capacity" column using the mean() method. This provided an insight into the general memory configuration of the smartphones in the dataset.
Q6: Check for missing values in the dataset
Explanation:
We used the isnull().sum() method to detect missing values in the dataset. This method returns the count of NaN (missing) values for each column, which helps in understanding the completeness of the data. For example, we observed that several columns had missing values, indicating areas for potential data cleaning or imputation.
Q7: Find the average RAM of all smartphones
Explanation:
The average RAM was calculated using the mean() function on the "ram_capacity" column. This gave us a numerical representation of the typical RAM capacity in the dataset, which was found to be approximately 6.56 GB.
Q8: Count the number of phones per brand
Explanation:
To count the number of phones per brand, we use the value_counts() method on the "Brand" column. This gives us the frequency of each unique brand in the dataset.
This line will output a count of how many phones belong to each brand, providing insight into the distribution of smartphone brands in the dataset.
Q9: Show the summary statistics for numeric columns
Explanation:
To display summary statistics (like mean, standard deviation, min, max, etc.) for the numeric columns, we use the describe() method.
This function will return a table of descriptive statistics for all numeric columns in the dataset, helping us understand the distribution and range of the numerical features such as price, rating, RAM, and more.
Q10: Which line would show you correlation between numeric columns?
Explanation:
To show the correlation between numeric columns, we use the corr() method.
This will calculate and display the correlation matrix between all numeric columns in the dataset. Values closer to 1 or -1 indicate a strong positive or negative correlation, respectively, while values near 0 indicate little or no correlation.
Q11: Import the required visualization libraries
Explanation:
To generate the necessary plots, we installed the seaborn library (if not already installed) and imported both matplotlib.pyplot and seaborn. These libraries are essential for creating advanced plots such as heatmaps, box plots, and scatter plots.
Q12: Plot a histogram of Price
Explanation:
To visualize the distribution of smartphone prices, we plotted a histogram using Seaborn or Matplotlib. A histogram is ideal for displaying the frequency of price ranges, helping to identify if most smartphones fall within a specific price range, and also revealing any skewness in the data.
Q13: Create a bar plot showing average price per brand
Explanation:
We used Seaborn's barplot() function to calculate and display the average price for each brand. This bar plot groups the data by brand and plots the mean price of smartphones for each brand, helping us understand the price distribution across different manufacturers.
Q14: Plot a heatmap of correlations
Explanation:
To visualize the correlation between various features in the dataset, we used Seaborn’s heatmap() function on the correlation matrix of the DataFrame. This heatmap allows us to easily spot which features have strong positive or negative correlations.
Q15: Show the relationship between RAM and Price using a scatter plot
Explanation:
We used Matplotlib’s scatter() function to plot the relationship between RAM capacity and price. This scatter plot shows individual data points and helps us visually inspect whether there's a correlation between smartphone RAM and price.
Q16: Create a boxplot to visualize Price distribution across brands
Explanation:
We used Seaborn’s boxplot() to display the price distribution for each smartphone brand. This boxplot shows the spread of prices, including the median, quartiles, and potential outliers for each brand.
Q17: Add a title to a plot
Explanation:
To add a title to any plot, we used Matplotlib’s plt.title() method. This function takes a string as input and adds it as the title to the plot, enhancing the clarity of the visual.
Q18: Plot a KDE (density plot) of prices
Explanation:
We plotted the Kernel Density Estimate (KDE) of the prices using Seaborn's kdeplot() function. This plot helps visualize the distribution of price values as a smooth curve, providing insights into the density of prices at various levels.
Q19: Rotate x-axis labels for readability
Explanation:
For better readability, especially when the x-axis labels are long, we used Matplotlib’s plt.xticks(rotation=45) to rotate the x-axis labels by 45 degrees. This makes the labels more readable when they are crowded.
Q20: Save a seaborn plot to a file
Explanation:
We saved the Seaborn plot to a file using plt.savefig(). This function allows you to export the plot as an image (e.g., PNG, JPEG), making it easy to share or embed in reports.
Q21: Import the required regression libraries
Explanation:
We imported essential libraries for performing regression tasks, including LinearRegression from sklearn.linear_model for linear regression, and train_test_split from sklearn.model_selection to split the dataset into training and testing sets. These libraries are crucial for building and evaluating regression models.
Q22: Define features X and target y
Explanation:
We defined the features X (independent variables) and the target variable y (dependent variable). For this case, we selected ram_capacity, internal_memory, and battery_capacity as features, and price as the target.
Q23: Split data into training and testing sets
Explanation:
We split the dataset into training and testing sets using train_test_split. This function takes X and y, along with a test size parameter (e.g., 0.2 for an 80/20 split), to ensure the model is trained on one portion and tested on another, which helps evaluate its performance.
Q24: Train a linear regression model
Explanation:
We trained a linear regression model using LinearRegression() from scikit-learn. This model was fitted to the training data (X_train, y_train), learning the relationship between the features and the target variable (price).
Q25: Make predictions on the test set
Explanation:
After training the model, we used the predict() method to make predictions on the test set (X_test). These predictions represent the estimated smartphone prices based on the features in the test data.
Q26: Calculate Mean Squared Error (MSE)
Explanation:
We calculated the Mean Squared Error (MSE) using the mean_squared_error() function from sklearn.metrics. MSE measures the average of the squares of the errors between the predicted and actual prices, providing an evaluation of the model's accuracy.
Q27: Print R² score
Explanation:
We printed the R² score of the model using the score() method. The R² score indicates how well the model's predictions match the actual data, with a value closer to 1 indicating better performance.
Q28: Add a new feature: Price_per_GB = Price / Storage
Explanation:
We created a new feature called Price_per_GB, which was calculated by dividing the price by the internal_memory (storage). This new feature helps analyze how the price of a smartphone relates to its storage capacity.
Q29: Scale features using StandardScaler
Explanation:
We scaled the features using StandardScaler from sklearn.preprocessing. This standardizes the features by removing the mean and scaling them to unit variance, which can improve the performance of certain machine learning models.
Q30: Try a Decision Tree Regressor
Explanation:
We experimented with a Decision Tree Regressor using DecisionTreeRegressor() from sklearn.tree. This model makes predictions by learning simple decision rules inferred from the data, and is particularly useful for non-linear relationships between features and target variables.
Section 4
Q31: Get feature importances from a trained decision tree
Explanation:
After training a Decision Tree Regressor, we extracted the feature importances using the feature_importances_ attribute. This attribute tells us which features (such as RAM, storage, etc.) are most influential in predicting the target variable (price) based on the decision tree model.
Q32: Plot feature importances
Explanation:
We plotted the feature importances using a bar plot to visualize which features contribute the most to the predictions of the decision tree model. This helps identify the most important features for predicting smartphone prices.
Q33: Select top 3 features using SelectKBest
Explanation:
We used SelectKBest from sklearn.feature_selection to select the top 3 features based on their scores. This technique ranks the features according to their performance in predicting the target variable and selects the most relevant ones for the model.
Q34: Show the score of each feature
Explanation:
After selecting the top features with SelectKBest, we examined their scores. These scores are calculated based on statistical tests that evaluate the importance of each feature in relation to the target variable, helping us identify the most predictive attributes.
Q35: Drop irrelevant columns like Model_Name
Explanation:
We removed irrelevant columns, such as Model_Name, from the dataset using the drop() function. This helps simplify the model by excluding features that don’t contribute to predicting the target variable.
Q36: Sort data by RAM descending
Explanation:
We sorted the dataset by the ram_capacity column in descending order using sort_values(). This helps prioritize phones with higher RAM, which could be more important for performance analysis.
Q37: Filter phones with RAM > 8GB and Battery > 4000mAh
Explanation:
We filtered the dataset to include only those smartphones with ram_capacity greater than 8GB and battery_capacity greater than 4000mAh. This is useful for analyzing high-performance smartphones with larger memory and better battery life.
Q38: Group by Brand and get the average of all numeric features
Explanation:
We grouped the dataset by brand_name and computed the mean of all numeric features for each brand using groupby() and mean(). This helps summarize how each brand performs across various features, such as price, RAM, and battery capacity.
Q39: Count how many phones have dual SIM support
Explanation:
We counted the number of smartphones that have dual SIM support by filtering the dataset where the has_dual_sim column is True and using sum() to calculate the total.
Q40: Create a pairplot to examine pairwise feature trends
Explanation:
We created a pairplot using Seaborn’s pairplot() function to examine pairwise relationships between selected features such as ram_capacity, internal_memory, battery_capacity, and price. This helps visualize correlations and trends among multiple features.

