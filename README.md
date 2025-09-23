# diploma
Master diploma

decoding - folder for decoding strategy

outputs - generated results

src - source code, schema, prompt

main.py - run dataset

test_query_model.py - test model

run_metrics - calculate metrics for strategy 


Metrics

Model: qwen

gready - 
Execution Accuracy: 174/500 = 0.3480
String Match Accuracy: 11/500 = 0.0220
Component Match Accuracy: 185.41666666666697/500 = 0.3708

beam -
Execution Accuracy: 196/500 = 0.3920
String Match Accuracy: 11/500 = 0.0220
Component Match Accuracy: 186.71666666666695/500 = 0.3734

top_k - 
Execution Accuracy: 171/500 = 0.3420
String Match Accuracy: 11/500 = 0.0220
Component Match Accuracy: 180.13333333333364/500 = 0.3603

top_p -
Execution Accuracy: 176/500 = 0.3520
String Match Accuracy: 11/500 = 0.0220
Component Match Accuracy: 180.05000000000032/500 = 0.3601

Custom beam -
Execution Accuracy: 139/500 = 0.2780
String Match Accuracy: 13/500 = 0.0260
Component Match Accuracy: 147.98333333333343/500 = 0.2960

EG beam (table, column check) -

Execution Accuracy: 144/500 = 0.2880
String Match Accuracy: 13/500 = 0.0260
Component Match Accuracy: 132.84999999999994/500 = 0.2657