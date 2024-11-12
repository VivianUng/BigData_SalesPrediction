import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder


# Function to load data
def load_data(file):
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
        return None
    except pd.errors.ParserError:
        st.error(
            "There was an issue parsing the file. Please check that it is a "
            "properly formatted CSV.")
        return None
    return df


def validate_data(df):
    required_columns = ['InvoiceId', 'Date', 'CustomerId', 'ProductId',
                        'Quantity', 'Amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return False
    return True


def preprocess_data(df):
    try:
        df.drop_duplicates(inplace=True)  # Remove duplicate rows

        # Remove negative values
        df = df[
            df['Quantity'] >= 1]  # Keep only rows with non-negative quantities
        df = df[df['Amount'] >= 1]  # Keep only rows with non-negative amounts

        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')

        # Sort the DataFrame by date in ascending order
        df.sort_values(by='Date', inplace=True)

        # Extract year, month, and quarter from the date
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter

        df.dropna(subset=['ProductId', 'Quantity', 'Amount'],
                  inplace=True)  # Remove rows with missing values in
        # specified columns

    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        return None
    return df


# Function to combine orders
def combine_orders_within_window(df, time_window_days):
    # Sort by 'CustomerId' and 'Date'
    df = df.sort_values(by=['CustomerId', 'Date'])

    combined_orders = []

    for customer_id, group in df.groupby('CustomerId'):
        group = group.reset_index(drop=True)
        start_idx = 0

        while start_idx < len(group):
            end_idx = start_idx + 1

            # Find end of window
            while end_idx < len(group) and (
                    group.loc[end_idx, 'Date'] - group.loc[
                start_idx, 'Date']).days <= time_window_days:
                end_idx += 1

            # Combine orders in the window
            window_orders = group.iloc[start_idx:end_idx]
            combined_order = window_orders.copy()
            combined_order = combined_order.groupby('ProductId').agg({
                'Quantity': 'sum',
                'Amount': 'sum'
            }).reset_index()

            # Add the customer ID and the start date of the window
            combined_order['CustomerId'] = customer_id
            combined_order['StartDate'] = group.loc[start_idx, 'Date']
            combined_order['Year'] = group.loc[start_idx, 'Year']  # Add Year
            combined_order['Month'] = group.loc[start_idx, 'Month']  # Add Month
            combined_orders.append(combined_order)

            start_idx = end_idx
            # Concatenate all combined orders
    result_df = pd.concat(combined_orders, ignore_index=True)

    return result_df


# EDA Functions
def eda_summary(df):
    st.subheader("Data Summary")
    st.write(df.head())
    st.write("Summary Statistics:")
    st.write(df.describe())
    st.write("Shape of the DataFrame:", df.shape)


def plot_top_products(df):
    try:
        col1, col2 = st.columns([3, 2])

        with col1:
            num_products = st.slider(
                "Select the number of top products to display", min_value=1,
                max_value=20, value=10)
            product_sales = df.groupby('ProductId').agg(
                {'Quantity': 'sum', 'Amount': 'sum'}).reset_index()
            top_products = product_sales.sort_values(by='Quantity',
                                                     ascending=False).head(
                num_products)

            if top_products.empty:
                st.warning("No data available for top products.")
                return

            plt.figure(figsize=(6, 4))  # Smaller chart size
            plt.bar(top_products['ProductId'], top_products['Quantity'],
                    color='skyblue')
            plt.title(f'Top {num_products} Popular Products by Quantity Sold')
            plt.xlabel('Product ID')
            plt.ylabel('Total Quantity Sold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")


def plot_top_customers(df):
    try:
        le_cust = LabelEncoder()
        df['CustomerId'] = le_cust.fit_transform(df['CustomerId'])
        customer_mapping = pd.DataFrame({'OriginalCustomerId': le_cust.classes_,
                                         'EncodedCustomerId': range(
                                             len(le_cust.classes_))})
        top_customer = df.groupby('CustomerId')['Amount'].sum().reset_index(
            name='Amount').sort_values(by='Amount', ascending=False).head(10)
        top_customer = top_customer.merge(customer_mapping,
                                          left_on='CustomerId',
                                          right_on='EncodedCustomerId',
                                          how='left')

        col1, col2 = st.columns([3, 2])

        with col1:
            # Add a slider to select the number of top customers
            num_customers = st.slider(
                "Select the number of top customers to display", min_value=1,
                max_value=20, value=10)

            top_customer = df.groupby('CustomerId')['Amount'].sum().reset_index(
                name='Amount').sort_values(by='Amount', ascending=False).head(
                num_customers)
            top_customer = top_customer.merge(customer_mapping,
                                              left_on='CustomerId',
                                              right_on='EncodedCustomerId',
                                              how='left')

            if top_customer.empty:
                st.warning("No data available for top customers.")
                return

            plt.figure(figsize=(6, 4))
            plt.bar(top_customer['OriginalCustomerId'], top_customer['Amount'],
                    color='skyblue')
            plt.title(f'Top {num_customers} Customers by Amount')
            plt.xlabel('Customer ID')
            plt.ylabel('Total Revenue')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")


def plot_sales_trend(df):
    try:
        col1, col2 = st.columns([3, 2])

        with col1:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()

            # for slider :
            # date_range = st.slider("Select the date range for sales trend",
            #                        min_value=min_date, max_value=max_date,
            #                        value=(min_date, max_date),
            #                        format="MM/YYYY")
            # filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (
            #             df['Date'] <= pd.to_datetime(date_range[1]))]

            start_date = st.date_input("Select the start date for sales trend",
                                       min_value=min_date, max_value=max_date,
                                       value=min_date)
            end_date = st.date_input("Select the end date for sales trend",
                                     min_value=min_date, max_value=max_date,
                                     value=max_date)

            if end_date < start_date:
                st.write("End date must be after start date")
            else:
                # Filter the dataframe based on the selected date range
                filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (
                        df['Date'] <= pd.to_datetime(end_date))]

                monthly_sales = \
                filtered_df.groupby(pd.Grouper(key='Date', freq='MS'))[
                    'Amount'].sum()

                plt.figure(figsize=(6, 4))
                monthly_sales.plot(kind='line')
                plt.title('Sales Trend Over Time')
                plt.xlabel('Month')
                plt.ylabel('Sales Amount')
                plt.xticks(rotation=45)
                st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")


def plot_monthly_sales(df):
    try:

        col1, col2 = st.columns([3, 2])

        with col1:
            years = sorted(df['Year'].unique())
            selected_years = st.multiselect("Select years to compare", years,
                                            default=years)
            monthly_sales = df[df['Year'].isin(selected_years)].groupby(
                ['Year', 'Month']).agg({'Amount': 'sum'}).reset_index()
            monthly_sales_pivot = monthly_sales.pivot(index='Month',
                                                      columns='Year',
                                                      values='Amount')

            plt.figure(figsize=(6, 4))
            for year in selected_years:
                if year in monthly_sales_pivot:
                    plt.plot(monthly_sales_pivot.index,
                             monthly_sales_pivot[year], marker='o', label=year)

            plt.title('Monthly Sales Comparison for Selected Years')
            plt.xlabel('Month')
            plt.ylabel('Total Sales (Amount)')
            plt.xticks(monthly_sales_pivot.index,
                       ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                        'Sep', 'Oct', 'Nov', 'Dec'])
            plt.legend(title='Year')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")


# Function to identify years in the dataset and dynamically split data
def split_data_by_year(df):
    years = sorted(df['Year'].unique())

    if len(years) < 3:
        st.warning(
            "Insufficient data for a complete training and test split. At "
            "least three years of data are recommended.")
        return None, None, None, None

    # Define training as all years except the latest one
    train_years = years[:-1]
    test_year = years[-1]

    le_cust = LabelEncoder()
    le_prod = LabelEncoder()
    df['CustomerId'] = le_cust.fit_transform(df['CustomerId'])
    df['ProductId'] = le_prod.fit_transform(df['ProductId'])

    X = df[['CustomerId', 'ProductId', 'Year', 'Month', 'Quantity']]
    y = df['Amount']

    X_train = X[df['Year'].isin(train_years)]
    y_train = y[df['Year'].isin(train_years)]
    X_test = X[df['Year'] == test_year]
    y_test = y[df['Year'] == test_year]

    return X_train, y_train, X_test, y_test


# Train the prediction model
def train_model(X_train, y_train, X_test, y_test):
    try:
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Prediction on training data
        training_data_prediction = model.predict(X_train)

        # R squared value
        r2_train = metrics.r2_score(y_train, training_data_prediction)

        # Calculate RMSE for training data
        rmse_train = np.sqrt(
            mean_squared_error(y_train, training_data_prediction))

        # Calculate MAE for training data
        mae_train = mean_absolute_error(y_train, training_data_prediction)

        # Calculate MAPE for training data
        mape_train = mean_absolute_percentage_error(y_train,
                                                    training_data_prediction)

        # Prediction on testing data
        test_data_prediction = model.predict(X_test)

        # R squared value
        r2_test = metrics.r2_score(y_test, test_data_prediction)

        # Calculate RMSE for test data
        rmse_test = np.sqrt(mean_squared_error(y_test, test_data_prediction))

        # Calculate MAE for test data
        mae_test = mean_absolute_error(y_test, test_data_prediction)

        # Calculate MAPE for test data
        mape_test = mean_absolute_percentage_error(y_test, test_data_prediction)

        # Store the results in a dictionary for easy display or logging
        model_evaluation = {
            'r2_train': r2_train,
            'rmse_train': rmse_train,
            'mae_train': mae_train,
            'mape_train': mape_train,
            'r2_test': r2_test,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'mape_test': mape_test
        }

        train_results = pd.DataFrame({
            'Month': X_train['Month'].values,
            'Actual': y_train,
            'Predicted': training_data_prediction
        })

        test_results = pd.DataFrame({
            'Month': X_test['Month'].values,
            'Actual': y_test,
            'Predicted': test_data_prediction
        })

        display_model_evaluation(model_evaluation)

        return train_results, test_results

    except ValueError as e:
        st.error(
            f"ValueError during model training: {e}. Please ensure the input "
            f"data is valid.")
    except Exception as e:
        st.error(f"An error occurred while training the model: {e}")
        return None, None


def display_model_evaluation(model_evaluation):
    st.subheader("Model Evaluation Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['R2 Score', 'RMSE', 'MAE', 'MAPE'],
        'Training': [model_evaluation['r2_train'],
                     model_evaluation['rmse_train'],
                     model_evaluation['mae_train'],
                     model_evaluation['mape_train']],
        'Testing': [model_evaluation['r2_test'], model_evaluation['rmse_test'],
                    model_evaluation['mae_test'], model_evaluation['mape_test']]
    })
    st.dataframe(metrics_df)


def plot_prediction_result(train_results, test_results):
    col1, col2 = st.columns([4, 2])

    with col1:
        # Aggregate by month for training data
        train_monthly = train_results.groupby('Month').agg(
            {'Actual': 'sum', 'Predicted': 'sum'}).reset_index()

        # Aggregate by month for test data
        test_monthly = test_results.groupby('Month').agg(
            {'Actual': 'sum', 'Predicted': 'sum'}).reset_index()

        # Calculate percentage variance
        test_monthly['Variance (%)'] = ((test_monthly['Predicted'] -
                                         test_monthly['Actual']) / test_monthly[
                                            'Actual']) * 100

        # Define the month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
                  'Oct', 'Nov', 'Dec']

        # Remove the index column for display
        test_monthly_display = test_monthly.set_index('Month', drop=True)

        # Display the DataFrame
        st.subheader("Monthly Aggregated Actual vs Predicted Amounts")

        st.dataframe(test_monthly_display.style.format(
            {'Actual': "{:.2f}", 'Predicted': "{:.2f}",
             'Variance (%)': "{:.2f}%"}))

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(months, test_monthly['Predicted'], color='skyblue',
                label='Predicted Amount')
        plt.title('Predicted Sales Amount (Monthly Aggregates)')
        plt.xlabel('Month')
        plt.ylabel('Predicted Amount')
        plt.legend()
        plt.grid(axis='y', linestyle='--',
                 alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

        # Plotting (Actual vs Predicted)
        plt.figure(figsize=(14, 6))

        # Train data Plot
        plt.plot(train_monthly['Month'], train_monthly['Actual'],
                 label='Actual (Train)', color='blue', marker='o')
        plt.plot(train_monthly['Month'], train_monthly['Predicted'],
                 label='Predicted (Train)', color='orange', marker='o')

        # Test data plot
        plt.plot(test_monthly['Month'], test_monthly['Actual'],
                 label='Actual (Test)', color='green', marker='o')
        plt.plot(test_monthly['Month'], test_monthly['Predicted'],
                 label='Predicted (Test)', color='red', marker='o')

        # Customize the plot
        plt.title('Actual vs Predicted Amounts by Month')
        plt.xlabel('Month')
        plt.ylabel('Amount')
        plt.xticks(range(1, 13), labels=months)
        plt.legend()
        plt.grid()

        # Show the plot
        plt.tight_layout()
        st.pyplot(plt)


def main():
    st.set_page_config(layout="wide")
    st.title("Sales Prediction Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    st.write(
        "The uploaded file should contain the following columns: InvoiceId, "
        "Date, CustomerId, ProductId, Quantity, Amount")

    if uploaded_file is not None:
        # Load data with improved error handling
        df = load_data(uploaded_file)
        if df is not None and validate_data(df):
            df = preprocess_data(df)
            if df is not None:  # Check if preprocessing was successful
                tab1, tab2, tab3 = st.tabs(
                    ["Uploaded data", "Data Analysis", "Sales Prediction"])

                with tab1:
                    st.header("Uploaded dataset")
                    st.dataframe(df)

                with tab2:
                    st.header("Exploratory Data Analysis")
                    eda_summary(df)
                    # Add a small divider
                    st.markdown(
                        "<hr style='border: 1px solid #B0B0B0; margin-bottom: "
                        "80px; margin-top: 20px;'>",
                        unsafe_allow_html=True)
                    st.subheader("Top Products")
                    plot_top_products(df)
                    st.markdown(
                        "<hr style='border: 1px solid #B0B0B0; margin-bottom: "
                        "80px; margin-top: 20px;'>",
                        unsafe_allow_html=True)
                    st.subheader("Top Customers")
                    plot_top_customers(df)
                    st.markdown(
                        "<hr style='border: 1px solid #B0B0B0; margin-bottom: "
                        "80px; margin-top: 20px;'>",
                        unsafe_allow_html=True)
                    st.subheader("Sales Trend")
                    plot_sales_trend(df)
                    st.markdown(
                        "<hr style='border: 1px solid #B0B0B0; margin-bottom: "
                        "80px; margin-top: 20px;'>",
                        unsafe_allow_html=True)
                    st.subheader("Monthly Sales")
                    plot_monthly_sales(df)

                with tab3:
                    st.header("Sales Prediction (Model : XGBoost)")
                    X_train, y_train, X_test, y_test = split_data_by_year(df)
                    if (X_train is not None and
                            y_train is not None and
                            X_test is not None and
                            y_test is not None):
                        if st.button("Run Prediction Model"):
                            train_results, test_results = train_model(X_train,
                                                                      y_train,
                                                                      X_test,
                                                                      y_test)
                            st.subheader("Predictions for the Test Year")
                            plot_prediction_result(train_results, test_results)
            else:
                st.error(
                    "Data preprocessing failed. Please check your dataset and "
                    "try again.")
    else:
        st.warning("Please upload a CSV file to get started.")


# Run the app
if __name__ == "__main__":
    main()
