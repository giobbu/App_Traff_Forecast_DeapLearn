FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]

EXPOSE 8501

# COPY . .
COPY . .

# CMD streamlit run run_.py

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "app", "streamlit", "run", "run_.py"]