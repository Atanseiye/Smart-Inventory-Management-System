try:
    import flask
    import streamlit as st
    import pandas as pd
except ModuleNotFoundError as e:
    print(f'{e} in the main.py')


from agents import inputAgent, prepare_sales_data, modelAgent, predict_demand, actionAgent, decisionMakingAgent

data = inputAgent()
sales_data_prep = prepare_sales_data(data['sales_data'])

# training the Agent
model = modelAgent(sales_data=sales_data_prep)

# making prediction
sales_predictions = predict_demand(model, data['sales_data'])

decisionResult = decisionMakingAgent(data['inventory_data'], data['sales_data'], data['budget'])

# print(decisionResult)

def dashboard_with_predictions(inventory, sales, decisions, priorityTable):
    st.title("Smart Inventory Management Dashboard")
    
    # Inventory Overview
    st.subheader("Inventory Overview")
    st.dataframe(inventory)
    
    # Sales Overview
    st.subheader("Sales Overview")
    st.dataframe(sales)
    
    # Restocking Decisions
    st.subheader("Restocking Decisions")
    decisions_df = pd.DataFrame(decisions)
    st.dataframe(decisions_df)

    # Priority Table
    st.subheader("Priority Table")
    prioritytable = pd.DataFrame(priorityTable)
    st.dataframe(prioritytable)
    
    # Visualizations
    st.subheader("Predicted Demand vs Current Stock")
    demand_vs_stock = sales[["product", "predicted_demand"]].merge(
        inventory[["product", "current_stock"]], on="product"
    )
    st.bar_chart(demand_vs_stock.set_index("product"))

dashboard_with_predictions(data["inventory_data"], data["sales_data"], decisionResult[0], decisionResult[1])