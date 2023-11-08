import matplotlib.pyplot as plt

def skillPlotter(finalData):
    skills = {}
    test = []
    #print(finalData)
    for i in finalData:
            
        
        for j in i['results']['full_matches']:
                if j['doc_node_value'] !='cancer':
                    
                    word = j['doc_node_value']              
                
                    if j['score']==1  :
                        if j['doc_node_value'] not in skills.keys():
                            
                            skills[j['doc_node_value']]=1
                        else:
                            skills[j['doc_node_value']]+=1
                
        for t in i['results']['ngram_scored']:
            word = t['doc_node_value'] 
          
            if t['score']==1:
                    if t['doc_node_value']!='cancer':
                        if t['doc_node_value'] not in skills.keys():
                            
                            skills[t['doc_node_value']]=1
                        else:
                            skills[t['doc_node_value']]+=1
            
    newSkills = {key: value for key, value in skills.items()}  
    return newSkills          
   