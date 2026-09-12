import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

products = ["Laptop", "Mobile", "Tablet", "Headphones", "Keyboard", "Mouse", "Monitor"]
categories = ["Electronics", "Accessories"]
cities = ["Latur", "Pune", "Mumbai", "Nashik", "Nanded", "Aurangabad"]
payment_modes = ["UPI", "Cash", "Credit Card", "Debit Card"]

df = pd.DataFrame({
    "Order_ID": [f"ORD{i:05d}" for i in range(1, n + 1)],
    "Order_Date": pd.date_range("2024-01-01", periods=n, freq="D"),
    "Customer_ID": [f"CUST{np.random.randint(1, 1001):04d}" for _ in range(n)],
    "Customer_Name": [f"Customer_{np.random.randint(1, 1001)}" for _ in range(n)],
    "Product": np.random.choice(products, n),
    "Category": np.random.choice(categories, n),
    "Quantity": np.random.randint(1, 10, n),
    "Unit_Price": np.random.randint(500, 80000, n),
    "City": np.random.choice(cities, n),
    "Payment_Mode": np.random.choice(payment_modes, n)
})

df["Sales"] = df["Quantity"] * df["Unit_Price"]

df.to_csv("data/sales_data.csv", index=False)

print("5000 sales records generated successfully!")
print("File: data/sales_data.csv")