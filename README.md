
# Deap Learning Traffic Forecasting
<img width="400" height="400" alt="model" src="https://user-images.githubusercontent.com/32134829/131836867-e37112fb-6748-4ca2-96d9-cfb0b432762c.png">


# Traffic Forecasting Streamlit App

<img width="727" alt="next_pred" src="https://user-images.githubusercontent.com/32134829/131837564-3665796c-30ba-429c-8edd-53abe8e49049.png">
<img width="858" alt="multi_pred" src="https://user-images.githubusercontent.com/32134829/131837568-154c5d0a-3498-4f3c-a907-e8cb514662ef.png">


----------
<img width="792" alt="multi_metr" src="https://user-images.githubusercontent.com/32134829/131840544-76df625e-e0ab-4770-bea5-a2ae637f4ad3.png">

### RMSE (left) and MAE (right)

----------
<img width="731" alt="total_metr" src="https://user-images.githubusercontent.com/32134829/131841102-e152dede-2ed0-4154-ab74-532f4236b2ac.png">

### dl - Deep Learning , nv - Naive Model 

## Build Image from Dockerfile

To create an image first:

```{r}
git clone https://github.com/giobbu/App_Traff_Forecast_DeapLearn 
cd App_Traff_Forecast_DeapLearn 
```

Then run:
```{r}
docker build -t giobbu/deap_traff_app .
```

Check the image created:
```{r}
docker image list
```

## Pull Image from Docker Hub
To downloaded the image from Docker Hub:
```{r}
docker pull  giobbu/deap_traff_app .
```

## Run Streamlit App Container
To interact with the App type:
```{r}
docker run -p 8501:8501 --rm giobbu/deap_traff_app
```
view your Streamlit app in your browser
```{r}
localhost:8501
```


