FROM jupyter/datascience-notebook

# Istall aditonal Python libraries
RUN pip install psycopg2-binary
RUN pip install --upgrade 'sqlalchemy<2.0'
RUN pip install pandas_datareader
RUN pip install setuptools==58
RUN pip install pygooglenews==0.1.2
RUN pip install feedparser==6.0.0
RUN pip install torch
RUN pip install textblob
RUN pip install tensorflow
RUN pip install pandas numpy matplotlib scikit-learn tensorflow
RUN pip install yfinance 
RUN pip install folium
RUN pip install requests


#Add Jupyter notebook and data
ADD aktienkurs-code.ipynb /home/jovyan
ADD apple-twitter-texts.csv /home/jovyan
