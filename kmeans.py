#1- Instancier le client Spark Session.
import findspark
findspark.init("C:/spark")
import pyspark 
findspark.find()

#importer des librairies
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd
import configparser


spark = SparkSession.builder \
                    .master("local") \
                    .appName("Kmeans") \
                    .getOrCreate()

# 2- Créer un fichier properties.conf contenant les informations relatives à vos paramètres du programme en dur.
fichier = open('properties.conf','w')
fichier.write("[Bristol-City-bike]\n")
fichier.write("Input-data=data/Bristol-city-bike.json\n")
fichier.write("Output-data=exported\n")
fichier.write("Kmeans-level=3")
fichier.close()

config = configparser.ConfigParser()
config.read("properties.conf")

path_to_input_data = config['Bristol-City-bike']['Input-data']
path_to_output_data = config['Bristol-City-bike']['Output-data']
num_partition_kmeans = int(config['Bristol-City-bike']['Kmeans-level']) 

# 3- Importer le json avec spark, en utilisant la variable path-to-input-data
brisbane = spark.read.json(path_to_input_data)
brisbane.show()

# 4-créer un nouveau data frame Kmeans-df contenant seulement les variables latitude et longitude.
Kmeansdf = brisbane.select("latitude","longitude")
Kmeansdf.show()

# 5- Kmeans
features = ( "longitude" ,"latitude")
kmeans = KMeans (). setK ( num_partition_kmeans ). setSeed ( 1 )
assembler = VectorAssembler ( inputCols = features , outputCol = "features" )
dataset = assembler . transform ( Kmeansdf )
model = kmeans.fit( dataset )
fitted = model.transform( dataset )

# 6- Les noms des colonnes de fitted.  
fitted.show()
fitted
# vérifier qu'Il s’agit bien de longitude, latitude, features et predictions.
fitted.columns

# 7- Déterminer les longitudes et latitudes moyennes pour chaque groupe en utilisant spark DSL et SQL.
#DSL
fitted.groupBy("prediction") \
    .agg(mean('latitude')\
    .alias('LatitudeMean'),mean('longitude')\
    .alias('LongitudeMean')\
        )\
    .show()

#SQL
# transformation du dataframe en table
fitted.createOrReplaceTempView("fittedSQL") 
spark.sql("""select prediction,
  mean(latitude) as LatitudeMean,
    mean(longitude) as LongitudeMean
    from fittedSQL
    group by prediction""").show()

 # les deux méthodes donnent les memes résultats

# 9- Elimination de la colonne features
NewData=fitted.drop("features")
NewData.show()    

# Exportation de la nouvelle data frame fitted dans le répertoire path-to-output-data

