# Analiza tektury i struktury memów i obrazów
Projekt wykonywany w ramach przedmiotu Analiza Obrazów i Wideo.
Celem projektu jest określanie tesktury i struktury memów i obrazów. Rozpoznawane są 4 klasy tekstury:
- zdjęcie,
- malowidło,
- kreskówka,
- tekst (np. screenshot czatu).

Jako strukturę rozumie się liczbę podobrazów, z~których składa się grafika. Prawidłowe działanie polega na wykryciu prawidłowej liczby podobrazów.
 

### Drzewo katalogów (wybrane foldery)

├── classifiers             # klasy zastosowanych klasyfikatorów 

│   ├── base_classifier.py  
│   ├── decision_trees.py  
│   ├── k_nearest_neighbours.py  
│   ├── naive_bayes.py  
│   ├── neural_network.py  
│   ├── tests_for_multiple_classifiers.py  
├── data                    # dane projektu
│   ├── base_dataset        # obrazy podzielone na klasy; jeden folder dla jednej klasy
│       ├── cartoon      
│       ├── painting  
│       ├── photo    
│       ├── text 
│       └── segmentation    # dane do segmentacji 
|   ├── combined_features_binaries  # pliki binarne z cechami wyliczonymi dla połączonego zbioru memów i obrazów z base_dataset
|   ├── memes_feature_binaries      # pliki binarne z cechami wyliczonymi dla memów z base_dataset
|   ├── pics_feature_binaries       # pliki binarne z cechami wyliczonymi dla obrazów z base_dataset
│   ├── results             # tu zapisywane są wszystkie metryki i obrazki do prezentacji i raportu
│       ├── metrics         # metryki z eksperymentów      
│       ├── plots           # wykresy, także zapisane w postaci graficznej tabelki z wynikami oraz macierze pomyłek  
│       ├── tables          # tabele z f1 score i accuracy 

├── data_loader             # funkcje do ładowanie danych do pamięci
│   ├── test_data_loader.py  
│   └── utils.py   

├── deep_learning            # klasy modeli i funkcji potrzebnych przy modelu głębokim
│   ├── dataset_loader.py   
│   ├── utils.py   
│   ├── vgg_model.py    
│   └── vgg_model_2.py  

├── experiments_scripts     # skrypty z badaniami oraz jupyter notebooki z wizualizacjami potrzebnymi w raporcie i prezentacjach
│   ├── classifier_based_on_classifiers.py 
│   ├── create_features.py
│   ├── create_pca_features.py 
│   ├── experiments_deep_learning.py 
│   ├── experiments_deep_learning_2.py
│   ├── experiments_for_multiple_classifiers.py 
│   ├── experiments_image_segmentation.py 
│   ├── experiments_on_dataset_size.py
│   ├── pandas_dataframes_results_with_latex.ipynb 
│   ├── results_deep_learning.ipynb 
│   ├── results_segmentation.ipynb 
│   └── test_binaries_shapes.py  
  
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




### do usuniecia
├── classifiers   
│   ├── base_classifier.py  
│   ├── decision_trees.py  
│   ├── k_nearest_neighbours.py  
│   ├── naive_bayes.py  
│   ├── neural_network.py  
│   ├── tests_for_multiple_classifiers.py  

├── data  
│   ├── base_dataset
│       ├── cartoon      
│       ├── painting  
│       ├── photo    
│       ├── text 
│       └── segmentation  # dane do segmentacji 
│   ├── rescaled_dataset *jeszcze nie zrobiony*
│   ├── results - *tu zapisywane są wszystkie metryki i obrazki do prezki*
│       ├── metrics *metrics from experiments*
│           ├── memes    
│           ├── pics       
│       ├── plots *plots contatining tables and confusion matrices*
│           ├── memes      
│           ├── pics    
│       ├── tables  *tables with average f1 score and accuracy*
│           ├── memes      
│           ├── pics  
|   ├── **memes_feature_binaries** *binarki memes z base_dataset*
|   ├── **pics_feature_binaries** *binarki pics z base_dataset*
│   ├── **datasets_pkl** - *tu zapisywane są wyliczone cechy w .pkl*

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
