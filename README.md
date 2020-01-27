# Analiza tektury i struktury memów i obrazów
Projekt wykonywany w ramach przedmiotu Analiza Obrazów i Wideo.
Celem projektu jest określanie tesktury i struktury memów i obrazów. Rozpoznawane są 4 klasy tekstury:
- zdjęcie,
- malowidło,
- kreskówka,
- tekst (np. screenshot czatu).

Jako strukturę rozumie się liczbę podobrazów, z~których składa się grafika. Prawidłowe działanie polega na wykryciu prawidłowej liczby podobrazów.
 
 
### Drzewo katalogów
```
├── classifiers             # klasy zastosowanych klasyfikatorów 

│   ├── base_classifier.py  
│   ├── decision_trees.py  
│   ├── k_nearest_neighbours.py  
│   ├── naive_bayes.py  
│   ├── neural_network.py  
│   ├── tests_for_multiple_classifiers.py  

├── data                    # dane projektu     
│   ├── base_dataset        # obrazy podzielone na klasy; jeden folder dla jednej klasy     
│      ├── cartoon         
│      ├── painting  
│      ├── photo    
│      ├── text        
│      ├── segmentation    # dane do segmentacji       
│   ├── combined_features_binaries  # pliki binarne z cechami wyliczonymi dla połączonego zbioru memów i obrazów   
│   ├── memes_feature_binaries      # pliki binarne z cechami wyliczonymi dla memów z base_dataset      
│   ├── pics_feature_binaries       # pliki binarne z cechami wyliczonymi dla obrazów z base_dataset        
│   ├── results             # tu zapisywane są wszystkie metryki i obrazki do prezentacji i raportu         
│      ├── metrics         # metryki z eksperymentów           
│      ├── plots           # wykresy, także zapisane w postaci graficznej tabelki z wynikami oraz macierze pomyłek     
│      ├── tables          # tabele z f1 score i accuracy  

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
  
├── feature_extraction       # klasy ze zdefiniowanymi ekstraktorami cech   
│   ├── bilateral_filter.py     
│   ├── color_counter.py  
│   ├── edges_detector.py  
│   ├── feature_namer.py    
│   ├── gabor_filter.py     
│   ├── hsv_analyser.py     
│   ├── kmeans_segmentator.py   
│   ├── test_bilateralFilter.py     
│   ├── test_colorCounter.py    
│   ├── test_edgesDetector.py   
│   ├── test_gabor_filter.py    
│   ├── test_hsvAnalyser.py 

├── feature_selection    # klasa służąca do tworzenia cech numerycznych      
│   └── feature_selector.py 

├── image_segmentation  # moduł dotyczący dzielenia obrazków na segmenty (wykrywania struktury)  
│   ├── hough_lines.py  
│   ├── hough_lines_results.ipynb   
│   └── test_houghLines.py  

├── visualization       # funkcje pomocnicze przy wizualizacji   
│   ├── images_size_plots.py    
│   ├── metrics_plots.py    
│   ├── multiple_image_plotter.py   
│   ├── single_image_plotter.py     
│   └── utils.py        

├── README.md       
├── settings.py         # importer ścieżek z user_settings.py   
└── user_settings.py    # konfiguracja potrzebnych ścieżek do danych i projektu     
```

#### Informacje dodatkowe

- Projekt wykonano w języku Python 3.6. Podczas implementacji korzystano ze środowiska PyCharm, dlatego zaleca się stosowanie tego środowiska.
Gwarantuje ono łatwość uruchomienia poszczególnych skryptów.

- Pliki z przedrostkiem "test_" zawierają testy jednostkowe implementowanych metod i funkcji.

- Pliki wykonywalne znajdują się w folderze `experiments_scripts`.

- Eksperymenty dotyczące rozpoznawania tekstury miały trzy główne fazy:
    -   wygenerowanie cech numerycznych i zapisanie ich do pliku,
    -   wygenerowanie zestawów cech, czyli połączenie cech z poprzedniego kroku w dłuższe wektory,
    -   przeprowadzenie eksperymentów na zapisanych zestawach cech.
    
- Eksperymenty dotyczące wykrywania struktury obrazów zawierają się w jednym pliku `experiment_scripts/experiments_image_segmentation.py`

### Instrukcja uruchomienia

1. Należy zdefiniować zmienne `DATA_PATH` oraz `PROJECT_PATH` w pliku `user_settings.py`.
`DATA_PATH` powinnien prowadzić do katalogu data, natomiast `PROJEKT_PATH` do katalogu głównego projektu (tego, w którym znajduje się plik `README.md`).
Proponowanym rozwiązaniem jest umieszczenie folderu `data` w katalogu `PROJECT_PATH`.

1. Projekt był rozwijany w środowisku PyCharm, dlatego proponuje się urochemie go w tym właśnie środowisku. 
Jako katalog projektu należy podać katalog główny (ten sam, do którego prowadzi ścieżka `PROJECT_PATH`).

1. Eksperymenty dotyczące tesktury powinny rozpocząć się od wygenerowania cech numerycznych. Można to wykonać uruchmiając plik `experiment_scripts/create_features.py`.
Skrypt przyjmuje trzy argumenty. Był on jako jedyny uruchamiany nie za pośrednictwem IDE, a prosto z terminala. 
Dzięki temu można było podzielić zbiór danych na memy i obrazy (przetworzyć je w osobnych terminalach) i w ten sposób przyspieszyć proces generacji cech. 
Poniżej przedstawino przykładowe polecenia, które należy wkleić do terminala otwartego w głównym katalogu projektu.
    ```
    python3 experiment_scripts/create_features.py --source-path "data/base_dataset" --type "pics" --dst-path "data/pics_feature_binaries"
    ```
    ```
    python3 experiment_scripts/create_features.py --source-path "data/base_dataset" --type "memes" --dst-path "data/memes_feature_binaries"
    ```
   Ważne, aby argument `dst-path` wskazywał na katalog w `DATA_PATH`. W przeciwnym wypadku nie zadziała dalsze przetwarzanie.
   
1. Następnie należy uruchomić plik `experiment_scripts/create_pca_features.py`, aby wygenerować cechy powstałe w wyniku redukcji wymiarowości. 
Ten plik można już uruchomić z poziomu PyCharma. 
Ładuje on pliki wygenerowane w poprzednim kroku i wykonuje na nich PCA. A następnie zapisuje do katalogów.

1. W następnym kroku można wykonać eksperymenty dotyczące tekstury, są to następujące pliki:
    1. `experiment_scripts/experiments_for_multiple_classifiers.py` - podstawowe badania dla wybranych zestawów cech na kilku klasyfikatorach.
    Badania są przeprowadzane zarówno dla memów, jak i obrazów.
    Skrypt sprawdza również wpływ standaryzacji, stosowania klasyfikatorów one_vs_rest oraz redukcji wymiarowości algorytmem PCA. 
    W części "main" sktyptu zdefiniowane są dwie listy `filenames_list` oraz `features_names`.
    Zawierają one definicję kolejnych zestawów cech oraz nazwy własne tych zestawów.
    Listy powinny mieć tę samą długość dla poprawnego działania.
    
    1. `experiment_scripts/experiments_on_dataset_size.py` - skrypt zawiera łączenie zbiorów memów i obrazów do nowego zbioru nazywanego "combined"
    oraz eksperymenty dotyczące wpływu rozmiaru zbioru danych.
    Plik zawiera podobne jak w przypadku `experiment_scripts/experiments_for_multiple_classifiers.py` `filenames_list` oraz `features_names` decydujące o zakresie badań.
    
    1. `experiment_scripts/classifier_based_on_classifiers.py` - zawiera eksperymenty dotyczące stworzenia klasyfikatora działającego na wynikach klasyfikacji innych klasyfikatorów.
    
    1. `experiment_scripts/experiments_deep_learning.py` - eksperymenty dotyczące modeli uczenia głębokiego.
    
1. Plik `experiment_scripts/experiments_image_segmentation.py` zawiera eksperymenty dotyczące wykrywacji struktury (segmentów).

1. Pozostałe pliki są plikami typu jupyter notebook i służyły do przygotowywania wyników pod kątem prezentacji i raportu.
    
    
    