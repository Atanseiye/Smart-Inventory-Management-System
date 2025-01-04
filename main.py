try:
    import flask
    import streamlit
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

print(decisionResult)