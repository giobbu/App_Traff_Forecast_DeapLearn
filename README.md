
# Deap Learning Traffic Forecasting
<img width="400" height="400" alt="model" src="https://user-images.githubusercontent.com/32134829/131836867-e37112fb-6748-4ca2-96d9-cfb0b432762c.png">


# Traffic Forecasting Streamlit App

<img width="511" alt="Screenshot 2021-09-21 at 13 25 41" src="https://user-images.githubusercontent.com/32134829/134163195-0f975df8-7808-447b-834c-a476e30aba52.png">
<img width="439" alt="Screenshot 2021-09-21 at 13 26 11" src="https://user-images.githubusercontent.com/32134829/134163218-c9446793-c790-47dd-9a44-5a84abd6580c.png">
<img width="387" alt="Screenshot 2021-09-21 at 13 27 49" src="https://user-images.githubusercontent.com/32134829/134163237-7eaa762e-aa81-48ec-856a-525e43d5c105.png">
<img width="394" alt="Screenshot 2021-09-21 at 13 26 39" src="https://user-images.githubusercontent.com/32134829/134163247-90176eb4-f7e0-4f03-ac3e-4042de9396fc.png">

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


