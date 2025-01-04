try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score, mean_absolute_error
    import openai
    import smtplib
except ModuleNotFoundError as e:
    print(f'{e} in the agents.py file')

# ===== AGENTS =====
def inputAgent():
    # Load the data
    inventory_data = pd.read_csv('data/inventory_data.csv')
    sales_data = pd.read_csv('data/sales_data.csv')
    budget = 5000
    return {'inventory_data':inventory_data, 
            'sales_data':sales_data, 
            'budget':budget}

def prepare_sales_data(sales_df):
    sales_df["sales_trend"] = sales_df["monthly_sales"] * np.random.uniform(0.9, 1.1, len(sales_df))
    return sales_df

def modelAgent(sales_data):
    X = sales_data[["monthly_sales", "sales_trend"]]
    y = sales_data["monthly_sales"] * np.random.uniform(1.05, 1.15, len(sales_data))  # Simulate future demand
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    performance = mean_absolute_error(y_test, prediction)
    print(f"Model Trained - MAE: {performance:.2f}")
    return model

# Predict future demand
def predict_demand(model, sales_df):
    X = sales_df[["monthly_sales", "sales_trend"]]
    predicted_demand = model.predict(X)
    sales_df["predicted_demand"] = predicted_demand
    return sales_df

def decisionMakingAgent(inventory_data, sales_data, budget):
    inventory_data['profit_margin'] = inventory_data['unit_cost'] * np.random.uniform(1.2, 2.0, len(inventory_data))

    # Merge sales_data and inventory data
    combined_data = pd.merge(inventory_data, sales_data, on='product')


    # Calculate priority score
    combined_data["priority_score"] = (
        combined_data["profit_margin"] * 0.5 +  # Weight for profit margin
        combined_data["monthly_sales"] * 0.3 +  # Weight for sales velocity
        (combined_data["reorder_threshold"] - combined_data["current_stock"]) * 0.2  # Weight for stock criticality
    )

    combined_data = combined_data.sort_values(by='priority_score', ascending=False)

    decisions = []
    total_cost = 0

    # Restock Logic
    for _, row in combined_data.iterrows():
        if row['current_stock'] < row['reorder_threshold'] and total_cost + row['unit_cost'] <= budget:
            restock_quantity = int(row['predicted_demand'] - row['current_stock'])
            restock_cost = row['unit_cost'] * restock_quantity
            if restock_cost <= (total_cost + budget):
                total_cost += restock_cost

                decisions.append({
                    "product": row["product"],
                    "restock_quantity": restock_quantity,
                    "priority_score": row["priority_score"],
                    'restock_cost': total_cost
                })
    
    return decisions, combined_data.head()


def action_agent(decisions):
    for decision in decisions:
        return f"Restocking {decision['restock_quantity']} units of {decision['product']}."
