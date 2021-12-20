
# Deap Learning Model for Traffic Forecasting



# Demo video

[![image alt text](https://github.com/giobbu/App_Traff_Forecast_DeapLearn/blob/master/img/ECDEC.jpg?raw=true)](https://github.com/giobbu/App_Traff_Forecast_DeapLearn/blob/master/img/demo.mp4)

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


