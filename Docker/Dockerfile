FROM jupyter/datascience-notebook

# Istall aditonal Python libraries
RUN pip install psycopg2-binary
RUN pip install --upgrade 'sqlalchemy<2.0'
RUN pip install pandas_datareader
RUN pip install setuptools==58
RUN pip install pygooglenews==0.1.2
RUN pip install feedparser==6.0.0

#Add Jupyter notebook and data
ADD aktienkurs-code.ipynb /home/jovyan