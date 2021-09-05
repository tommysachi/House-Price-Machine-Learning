# House Price Prediction
The Main Objective of this EDA and Machine Learning is to provide a Property Developer company in order to to understanding the Data, so that they can increase sales by targeting market and launch precised product.

**1. Initial Hypothesis and EDA**
- Number of Bedrooms will affect the Price
- Number of Bathrooms will affect the Price
- Area sqft living and lot will affect the Price
- Total FLoors will affect the Price
- Waterfront View will affect the Price
- Total View will affect the Price
- Current Rating Condition will Affect the Price
- Grade will affect the Price
- Year Built will affect the Price
- Year Renovated will affect the Price

![Bedroom VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Bedroom%20VS%20Price.JPG)

![Bathroom VS Price](hhttps://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Bathroom%20VS%20Price.JPG)

![Area Living VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/SQFT%20Living%20VS%20Price.JPG)

![Area Lot VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/SQFT%20Lot%20VS%20Price.JPG)

![Waterfront VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Waterfront%20VS%20Price.JPG)

![View VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/View%20VS%20Price.JPG)

![Condition VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Condition%20VS%20Price.JPG)

![Grade VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Grade%20VS%20Price.JPG)

![Year Built VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Year%20Built%20VS%20Price.JPG)

![Year Renovated VS Price](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Year%20Renovated%20VS%20Price.JPG)

From these categories, we can see that there is correlation of these features that affect the house price.

**2. Machine Learning**

The Evaluation Matrix of the Algorithm that used:

- Linear Regression

    ![Linear Regression](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Evaluation%20Matrix%20Linear%20Regression.JPG)

- Ridge

    ![Ridge](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Ridge%20Evaluation%20Matrix.JPG)

- Lasso

    ![Lasso](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Evaluation%20Matrix%20Lasso.JPG)

- Elastic Net

    ![Elastic Net](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Evaluation%20Matrix%20Elastic%20Net.JPG)

**3. Optimization**

Using Polynomial Features and Power Transform, we try to optimize the Error of the Machine Learning Model.

```
poly = PolynomialFeatures(degree=2,include_bias=False)
yeo_pow = PowerTransformer(method = 'yeo-johnson')
num_pow = yeo_pow.fit_transform(num_poly)
df_pow = pd.DataFrame(num_pow)
```
![Linear Regression Opt](https://github.com/tommysachi/House-Price-Machine-Learning/blob/main/Tabel%20%26%20Visual/Evaluation%20Matrix%20Linear%20Regression%20(with%20Optimization).JPG)

After Optomization, we can increase the R2 Score, that means model was successfully optimized.



