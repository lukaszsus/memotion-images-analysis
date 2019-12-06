# Memotion-images-analysis
Academic project for subject Image and video analysis. It is a part of bigger project called Memotion Analysis.

---------------
### TODO

- [Ł] Zmergować kod
- [M] ~~Dodać MLP, inne klasyfikatory, cross_validation~~
    * zapisywać wyniki do pliku
- [Ł] Dodać zapisywanie cech
    * dodać wersję z przeskalowaniem (tam gdzie konieczne albo wszędzie)
- [M] ~~Dodać trochę danych treningowych i wysłać Łukaszowi~~
- [M] Dodać PCA i segmentację kMeansem
- [M] Dorobić wykresiki (inne niż *confussion matrix*)


### Drzewo katalogów (wybrane foldery)

├── classifiers   
│   ├── base_classifier.py  
│   ├── decision_trees.py  
│   ├── k_nearest_neighbours.py  
│   ├── naive_bayes.py  
│   ├── neural_network.py  
│   ├── tests_for_multiple_classifiers.py  

├── data  
│   ├── cartoon  
│   ├── **datasets_pkl** - *tu zapisywane są wyliczone cechy w .pkl*    
│   ├── painting  
│   ├── photo  
│   ├── **results** - *tu zapisywane są wszystkie metryki i obrazki do prezki*  
│   └── text  

├── **data_as_dataset_saver.py** - *tu jest plik do zapisywania cech na brudno (mma)*  

├── data_loader  
│   ├── test_data_loader.py  
│   └── utils.py   
  
├── feature_extraction  
│   ├── bilateral_filter.py  
│   ├── color_counter.py  
│   ├── edges_detector.py  
│   ├── feature_namer.py  
│   ├── hsv_analyser.py  
│   ├── kmeans_segmentator.py  

├── feature_selection  
│   ├── dataset_creator.py  
│   └── test_datasetCreator.py  

├── **results** - *przeniosłabym to do ./data/results do jakiegoś podfolderu może*  
│   ├── models  
│   ├── plots  
│   └── tables  
