from flask import Flask, render_template, jsonify, request
import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


app = Flask(__name__)


DATA_FILE = os.path.join(
    "data",
    "sales_data.csv"
)


def load_data():
    return pd.read_csv(DATA_FILE)


# =========================
# SALES FORECAST
# =========================

def predict_next_month_sales(df):

    df = df.copy()

    if df.empty:
        return {
            "month": "N/A",
            "predicted_sales": 0,
            "mae": 0,
            "r2_score": 0
        }

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"]
    )

    # Monthly sales
    monthly_sales = (
        df.groupby(
            df["Order_Date"].dt.to_period("M")
        )["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales = monthly_sales.sort_values(
        "Order_Date"
    ).reset_index(drop=True)

    if len(monthly_sales) < 13:
        return {
            "month": "N/A",
            "predicted_sales": 0,
            "mae": 0,
            "r2_score": 0
        }

    # =========================
    # TIME SERIES FEATURES
    # =========================

    monthly_sales["Month_Number"] = (
        range(1, len(monthly_sales) + 1)
    )

    monthly_sales["Month"] = (
        monthly_sales["Order_Date"]
        .dt.month
    )

    monthly_sales["Year"] = (
        monthly_sales["Order_Date"]
        .dt.year
    )

    # Lag features
    monthly_sales["Lag_1"] = (
        monthly_sales["Sales"].shift(1)
    )

    monthly_sales["Lag_3"] = (
        monthly_sales["Sales"].shift(3)
    )

    monthly_sales["Lag_6"] = (
        monthly_sales["Sales"].shift(6)
    )

    monthly_sales["Lag_12"] = (
        monthly_sales["Sales"].shift(12)
    )

    # Rolling averages
    monthly_sales["Rolling_3"] = (
        monthly_sales["Sales"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    monthly_sales["Rolling_6"] = (
        monthly_sales["Sales"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    model_data = monthly_sales.dropna().copy()

    if len(model_data) < 10:
        return {
            "month": "N/A",
            "predicted_sales": 0,
            "mae": 0,
            "r2_score": 0
        }

    features = [
        "Month_Number",
        "Month",
        "Year",
        "Lag_1",
        "Lag_3",
        "Lag_6",
        "Lag_12",
        "Rolling_3",
        "Rolling_6"
    ]

    X = model_data[features]
    y = model_data["Sales"]

    # =========================
    # TRAIN / TEST SPLIT
    # =========================

    test_size = max(
        6,
        int(len(model_data) * 0.2)
    )

    train_data = model_data.iloc[:-test_size]
    test_data = model_data.iloc[-test_size:]

    X_train = train_data[features]
    y_train = train_data["Sales"]

    X_test = test_data[features]
    y_test = test_data["Sales"]

    # =========================
    # RANDOM FOREST MODEL
    # =========================

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=3,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # Test predictions
    test_predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    r2 = r2_score(
        y_test,
        test_predictions
    )

    # =========================
    # TRAIN FINAL MODEL
    # =========================

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=3,
        random_state=42
    )

    final_model.fit(
        X,
        y
    )

    # =========================
    # NEXT MONTH FEATURES
    # =========================

    last_row = monthly_sales.iloc[-1]

    last_month = last_row["Order_Date"]

    next_month = last_month + 1

    next_month_number = (
        len(monthly_sales) + 1
    )

    recent_sales = (
        monthly_sales["Sales"]
        .tolist()
    )

    next_features = pd.DataFrame(
        [{
            "Month_Number":
                next_month_number,

            "Month":
                next_month.month,

            "Year":
                next_month.year,

            "Lag_1":
                recent_sales[-1],

            "Lag_3":
                recent_sales[-3],

            "Lag_6":
                recent_sales[-6],

            "Lag_12":
                recent_sales[-12],

            "Rolling_3":
                sum(recent_sales[-3:]) / 3,

            "Rolling_6":
                sum(recent_sales[-6:]) / 6
        }]
    )

    predicted_sales = final_model.predict(
        next_features[features]
    )[0]

    predicted_sales = max(
        0,
        float(predicted_sales)
    )

    return {
        "month": str(next_month),

        "predicted_sales":
            round(
                predicted_sales,
                2
            ),

        "mae":
            round(
                float(mae),
                2
            ),

        "r2_score":
            round(
                float(r2),
                4
            )
    }
    
# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================
# DASHBOARD API
# =========================

@app.route("/api/dashboard")
def dashboard():

    city = request.args.get("city")
    product = request.args.get("product")
    payment = request.args.get("payment")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    df = load_data()

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"]
    )


    # =========================
    # FILTERS
    # =========================

    if city and city != "All":

        df = df[
            df["City"] == city
        ]


    if product and product != "All":

        df = df[
            df["Product"] == product
        ]


    if payment and payment != "All":

        df = df[
            df["Payment_Mode"] == payment
        ]


    if start_date:

        df = df[
            df["Order_Date"]
            >= pd.to_datetime(start_date)
        ]


    if end_date:

        df = df[
            df["Order_Date"]
            <= pd.to_datetime(end_date)
        ]


    # =========================
    # KPI ANALYTICS
    # =========================

    total_revenue = float(
        df["Sales"].sum()
    )

    total_orders = int(
        df["Order_ID"].nunique()
    )

    total_customers = int(
        df["Customer_ID"].nunique()
    )

    if total_orders > 0:

        average_order_value = round(
            total_revenue /
            total_orders,
            2
        )

    else:

        average_order_value = 0


    # =========================
    # TOP PRODUCTS
    # =========================

    top_products = (
        df.groupby("Product")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    top_products_data = [

        {
            "product": str(index),
            "sales": round(
                float(value),
                2
            )
        }

        for index, value
        in top_products.items()

    ]


    # =========================
    # CITY SALES
    # =========================

    city_sales = (
        df.groupby("City")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    city_sales_data = [

        {
            "city": str(index),
            "sales": round(
                float(value),
                2
            )
        }

        for index, value
        in city_sales.items()

    ]


    # =========================
    # CATEGORY SALES
    # =========================

    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    category_sales_data = [

        {
            "category": str(index),
            "sales": round(
                float(value),
                2
            )
        }

        for index, value
        in category_sales.items()

    ]


    # =========================
    # MONTHLY SALES
    # =========================

    monthly_sales = (
        df.groupby(
            df["Order_Date"].dt.to_period("M")
        )["Sales"]
        .sum()
    )

    monthly_sales_data = [

        {
            "month": str(index),
            "sales": round(
                float(value),
                2
            )
        }

        for index, value
        in monthly_sales.items()

    ]


    # =========================
    # CUSTOMER ANALYTICS
    # =========================

    customer_sales = (
        df.groupby("Customer_ID")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    top_customers_data = [

        {
            "customer_id": str(index),
            "sales": round(
                float(value),
                2
            )
        }

        for index, value
        in customer_sales.items()

    ]


    customer_orders = (
        df.groupby("Customer_ID")["Order_ID"]
        .nunique()
    )

    repeat_customers = int(
        (customer_orders > 1).sum()
    )

    one_time_customers = int(
        (customer_orders == 1).sum()
    )


    if total_customers > 0:

        average_customer_spending = round(
            total_revenue /
            total_customers,
            2
        )

    else:

        average_customer_spending = 0


    # =========================
    # CUSTOMER SEGMENTATION
    # =========================

    customer_summary = (
        df.groupby("Customer_ID")["Sales"]
        .sum()
        .reset_index()
    )


    def segment_customer(sales):

        if sales >= 3000000:

            return "High Value"

        elif sales >= 1000000:

            return "Regular"

        else:

            return "Low Value"


    customer_summary["Segment"] = (
        customer_summary["Sales"]
        .apply(segment_customer)
    )


    customer_segments_data = (
        customer_summary
        .groupby("Segment")
        .size()
        .reset_index(
            name="Customers"
        )
    )


    customer_segments_data = [

        {
            "segment": str(
                row["Segment"]
            ),

            "customers": int(
                row["Customers"]
            )
        }

        for _, row
        in customer_segments_data.iterrows()

    ]


    # =========================
    # PRODUCT INSIGHTS
    # =========================

    if not df.empty:

        product_sales = (
            df.groupby("Product")["Sales"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        best_product = str(
            product_sales.index[0]
        )

        best_product_sales = round(
            float(product_sales.iloc[0]),
            2
        )

        lowest_product = str(
            product_sales.index[-1]
        )

        lowest_product_sales = round(
            float(product_sales.iloc[-1]),
            2
        )

    else:

        best_product = "N/A"
        best_product_sales = 0

        lowest_product = "N/A"
        lowest_product_sales = 0


    # =========================
    # HIGHEST SALES CITY
    # =========================

    if not df.empty:

        city_revenue = (
            df.groupby("City")["Sales"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        highest_city = str(
            city_revenue.index[0]
        )

        highest_city_sales = round(
            float(city_revenue.iloc[0]),
            2
        )

    else:

        highest_city = "N/A"
        highest_city_sales = 0


    # =========================
    # DYNAMIC FORECAST
    # =========================

    if len(df) >= 2:

        prediction = (
            predict_next_month_sales(df)
        )

    else:

        prediction = {

            "month": "N/A",

            "predicted_sales": 0,

            "mae": 0,

            "r2_score": 0
        }


    # =========================
    # FINAL RESPONSE
    # =========================

    return jsonify({

        "kpis": {

            "total_revenue":
                round(
                    total_revenue,
                    2
                ),

            "total_orders":
                total_orders,

            "total_customers":
                total_customers,

            "average_order_value":
                average_order_value
        },


        "top_products":
            top_products_data,


        "city_sales":
            city_sales_data,


        "category_sales":
            category_sales_data,


        "monthly_sales":
            monthly_sales_data,


        "prediction":
            prediction,


        "customer_analytics": {

            "top_customers":
                top_customers_data,

            "repeat_customers":
                repeat_customers,

            "one_time_customers":
                one_time_customers,

            "average_customer_spending":
                average_customer_spending
        },


        "customer_segmentation":
            customer_segments_data,


        "product_insights": {

            "best_product":
                best_product,

            "best_product_sales":
                best_product_sales,

            "lowest_product":
                lowest_product,

            "lowest_product_sales":
                lowest_product_sales,

            "highest_city":
                highest_city,

            "highest_city_sales":
                highest_city_sales
        }

    })


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":

    print(
        "Smart Sales Analytics starting..."
    )

    print(
        "Dataset:",
        DATA_FILE
    )

    app.run(
        debug=True
    )