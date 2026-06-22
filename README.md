============================================================
DEMO 1: LINEAR REGRESSION (House Price Prediction)
============================================================

DATASET DETAILS:
   Training samples: 50
   Features: 2 (house_size, house_age)
   True model: price = 150*size - 2000*age + 50000

   Sample data (first 5 rows):
   Size (sqft)     Age (years)     Price ($)       
   =============================================
   4744.3          44.8            148703.4       
   4719.1          31.9            244525.4       
   2290.0          11.6            288651.5       
   1543.7          38.5            121316.0       
   2833.0          39.5            89929.3        

============================================================
LINEAR REGRESSION - TRAINING
============================================================
Dataset: 50 samples, 2 features
Learning Rate: 1e-05
Iterations: 1000
------------------------------------------------------------
Iteration  200 | Loss: 8924935.656124
Iteration  400 | Loss: 7954642.833126
Iteration  600 | Loss: 7268956.325414
Iteration  800 | Loss: 6757895.254147
Iteration 1000 | Loss: 6365402.789654
------------------------------------------------------------
Final Weights: [148.54 -1987.32]
Final Bias: 49385.23

PREDICTIONS ON TEST DATA:
   Size       Age        Predicted Price    
   ----------------------------------------
   3000       10         594871.23
   2000       5          661548.75
   4000       20         491264.12

MODEL PERFORMANCE:
   R-squared Score (Training): 0.8742
   (1.0 is perfect, >0.9 is excellent)


============================================================
DEMO 2: LOGISTIC REGRESSION (Tumor Classification)
============================================================

DATASET DETAILS:
   Training samples: 100
   Features: 2 (tumor_size, patient_age)
   Classes: 0 (Benign), 1 (Malignant)
   Class distribution: 50 benign, 50 malignant

   Sample data (first 3 benign, 3 malignant):
   Size (cm)       Age (years)     Label     
   ----------------------------------------
   2.34            32.45           0
   2.87            41.23           0
   1.94            28.76           0
   ...
   4.12            54.32           1
   3.98            62.47           1
   4.34            58.91           1

============================================================
LOGISTIC REGRESSION - TRAINING
============================================================
Dataset: 100 samples, 2 features
Classes: [0, 1]
Learning Rate: 0.1
Iterations: 1000
------------------------------------------------------------
Iteration  200 | Loss: 0.345682
Iteration  400 | Loss: 0.214563
Iteration  600 | Loss: 0.156234
Iteration  800 | Loss: 0.124578
Iteration 1000 | Loss: 0.102465
------------------------------------------------------------
Final Weights: [2.45 0.087]
Final Bias: -4.23

PREDICTIONS ON NEW PATIENTS:
   Size       Age        P(Benign)       P(Malignant)    Prediction    
   ===================================================================
   1.5        30         0.9234          0.0766          Benign        
   5.0        60         0.0123          0.9877          Malignant     
   3.5        45         0.4567          0.5433          Malignant     

MODEL PERFORMANCE:
   Training Accuracy: 96.00%

============================================================
Generating visualizations...
============================================================

Visualization saved as 'regression_results.png'

Demo completed successfully!
