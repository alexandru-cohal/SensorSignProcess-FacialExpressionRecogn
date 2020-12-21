# SensorSignProcess-FacialExpressionRecogn
### The *"Facial Expressions Recognition Using the Emotiv EPOC Headset"* project developed for the *"Sensor Signal Processing"* course within the *EMECS* Masters

- **Date**: March 2018
- **Purpose**: The purpose of this project is to develop a program which recognizes in real-time the Neutral state and 5 facial expressions (left wink / blink, right wink / blink, strong blink, open mouth, full mouth) using the 14 channels data provided by the *Emotiv EPOC Headset*
- **Programming Language**: Python
- **Team**: 
  - Vitor Ribeiro Roriz
  - Alexandru Cohal
- **Inputs**:
  - The 14 channels data provided by the *Emotiv EPOC Headset* (monitoring the electrical activity of the muscle tissue)
- **Outputs**:
  - The facial state (neutral, left wink / blink, right wink / blink, strong blink, open mouth, full mouth)
- **Solution**:
  - The classical steps were followed: data acquisition, data preprocessing, feature extraction and classification 
  - A comparison between multiple calssifiers was performed: 
    - Decision Tree classifier based on thresholds
    - Multi-Layer Perceptron
    - Support Vector Machine
    - K-Nearest Neighbours
  - For more information about the solution, implementation, results, conclusions and improvements see [this document](documentation/SensorSignalProcessing-FacialExpressionsRecognition-Documentation.pdf)
- **Results**:
  - Real-Time classification of the previously specified facial expression was sucessful with a period of 0.07 seconds
  - The best classifier was K-Nearest Neighbours (K = 11)
  - For more information about the solution, implementation, results, conclusions and improvements see [this document](documentation/SensorSignalProcessing-FacialExpressionsRecognition-Documentation.pdf)
