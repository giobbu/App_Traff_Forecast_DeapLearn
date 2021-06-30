# Setup
The DL model benefits from GPU usage. You can have access to free GPUs using [Google Colab](#Colab).

## Local

To create an environment from a given `environment.yml` 
```{r}
cd development
conda env create --name envname --file=environment.yml
```

## Colab

* Go to https://colab.research.google.com, and upload `VSCode_Colab.ipynb`. 

* Connect your new notebook to a GPU runtime by doing `Runtime > Change Runtime type > GPU`.

#### Clone Repository in Colab
* Get a local copy of the repository to work on, and allow Python to find packages in the current working directory use:

```{r}
!git clone https://github.com/giobbu/traffic_enc_dec
%env PYTHONPATH=.:$PYTHONPATH
%cd traffic_enc_dec/development
%ls
```
Enter the  `traffic_enc_dec` directory and make sure things work:

```{r}
!python run_.py
```


#### VSCode on Google Colab

It is possible to use the VSCode interface in Colab.

* Run the cell with following command:

```{r}
!curl -fsSL https://code-server.dev/install.sh | sh
!pip install -qqq pyngrok

from pyngrok import ngrok
url = ngrok.connect(port=80) #9000
print(url)
!nohup code-server --port 80 --auth none & #9000
```

* Clicking the ngrok link takes you to a web VSCode interface. Go to `Open folder` and type  `/content/traffic_enc_dec/development`.

![Alt text](https://github.com/giobbu/traffic_enc_dec/blob/main/setup/colab.png)

* In a `New Terminal` use `PYTHONPATH=. python3 run_.py `.
