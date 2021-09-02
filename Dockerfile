FROM continuumio/miniconda3

WORKDIR /home/app

ADD environment.yml environment.yml
RUN conda env create -f environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

EXPOSE 8501

COPY . .

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "app_conda_deap", "streamlit", "run", "run_.py"]