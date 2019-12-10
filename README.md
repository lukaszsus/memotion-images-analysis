# Memotion-images-analysis
Academic project for subject Image and video analysis. It is a part of bigger project called Memotion Analysis.

---------------
## TODO

- PCA
- Skalowanie na zasadzie segmentacji obrazów 
  (segmentacja, ustandaryzowanie mody histogramu)
- Zespół klasyfikatorów
    - cechy klasyfikatorów jako wejście do sieci neuronowej
- Klasyfikacja z wyodrębnieniem jednej klasy
- Wykres skuteczności w zależności od wielkości zbioru danych
- ###Dzielenie danych na podobrazki (badanie struktury)
    - badanie gradientów wzdłuż pionów i poziomów obrazka
    - tworzenie bounding boxów

- [Ł] ~~Połączyć memes i pics i zbadać zależność wyników od rozmiaru~~
- [Ł] ~~Zrobić PCA~~
- [Ł] Zrobić klasyfikator złożony, który zrobi klasyfikację na podstawie klasyfikacji innych klasyfikatorów
- [Ł] ~~Zrobić filtry Gabora lub inną cechę częstotliwościową~~
- [Ł] ~~Zrobić klasyfikację 1-vs-rest~~
- [Ł] Porównać z uczeniem głębokim, np. VGG-16

- [Ł] Wygenerować sztuczny zbiór obrazów połączonych
- [M] Sprawdzić brutal force dla separacji obrazów połączonych - np. znaleźć gwałtowną zmianę gradientu



## Drzewo katalogów (wybrane foldery)

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
│       └── text  
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