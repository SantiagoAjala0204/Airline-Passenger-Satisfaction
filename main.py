from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("modelo_decision_tree_Airline_Satisfaction.pkl")  # Reemplaza con tu modelo Random Forest

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Age: int = Form(...),
    Flight_Distance: int = Form(...),
    Inflight_wifi_service: int = Form(...),
    Departure_Arrival_time_convenient: int = Form(...),
    Food_and_drink: int = Form(...),
    Seat_comfort: int = Form(...),
    Inflight_entertainment: int = Form(...),
    Departure_Delay_in_Minutes: int = Form(...),
    Arrival_Delay_in_Minutes: int = Form(...),
    Gender: str = Form(...),
    Customer_Type: str = Form(...),
    Type_of_Travel: str = Form(...),
    Class: str = Form(...)
):
    # Codificación manual de variables categóricas (One-Hot)
    gender_female = 1 if Gender == "Female" else 0
    gender_male = 1 if Gender == "Male" else 0

    customer_loyal = 1 if Customer_Type == "Loyal Customer" else 0
    customer_disloyal = 1 if Customer_Type == "disloyal Customer" else 0

    travel_business = 1 if Type_of_Travel == "Business travel" else 0
    travel_personal = 1 if Type_of_Travel == "Personal Travel" else 0

    class_business = 1 if Class == "Business" else 0
    class_eco = 1 if Class == "Eco" else 0
    class_eco_plus = 1 if Class == "Eco Plus" else 0

    features = np.array([[
        Age, Flight_Distance, Inflight_wifi_service, Departure_Arrival_time_convenient,
        Food_and_drink, Seat_comfort, Inflight_entertainment,
        Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes,
        gender_female, gender_male,
        customer_loyal, customer_disloyal,
        travel_business, travel_personal,
        class_business, class_eco, class_eco_plus
    ]])

    prediction = model.predict(features)[0]
    return templates.TemplateResponse("form.html", {"request": request, "result": f"Predicción: {prediction}"})
