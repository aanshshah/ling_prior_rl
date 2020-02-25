# WordTree

### Simple API for comparing hyper and hypon difference between two words

   - [Use BaseHTTPServer](#WordTreeByBaseHTTP.py)
     - WordTreeByBaseHTTP.py
   - [Use Flask](#WordTreeByFlask.py)
     - WordTreeByFlask.py
    
    
### Example
  - hypernyms are the synsets that are more general
    - 127.0.0.1:8080/hyper?word1=banana&word2=apple
    ![Screenshot](https://github.com/jamie2017/WordTree/blob/master/output/banana.n.02_vs_apple.n.01_hyper_.png)

  - hyponyms are the synsets that are more specific
    - 127.0.0.1:8080/hyper?word1=banana&word2=apple
    ![Screenshot](https://github.com/jamie2017/WordTree/blob/master/output/banana.n.02_vs_apple.n.01_hypon_.png)


### Requirement
    python==2.7
    pydot==1.2.3
    Flask==0.12.1
    nltk==3.2.2
    Werkzeug==0.12.1
### Notes
    New GET:
    @time: 4/21/17 4:31 PM
    1. Handle 404.html and image file
    2. Use pydot recursively draw node and generate plot
    3. Familiar with NLTK WordNet libary 
    4. Api practice with Flask and BaseHTTPServer
    


#### Reference:
http://www.dh2012.uni-hamburg.de/conference/programme/abstracts/automatic-topic-hierarchy-generation-using-wordnet.1.html


