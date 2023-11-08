
import PyPDF2
from gensim.models import FastText
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from SkillPlotter import skillPlotter
import spacy
from spacy.matcher import PhraseMatcher
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
import datetime
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
#nlp = spacy.load("en_core_web_lg")
#fasttextmodel = FastText.load("fasttextcbow_model.bin")


model_cache = {}

def load_fasttext_model():
    if 'fasttext' not in model_cache:
        # Load and initialize the FastText model
        model_cache['fasttext'] = FastText.load("fasttextcbow_model.bin")
    return model_cache['fasttext']

def load_spacy_model():
    if 'nlp' not in model_cache:
        # Load and initialize the spaCy model
        model_cache['nlp'] = spacy.load("en_core_web_lg")
    return model_cache['nlp']
fasttextmodel = load_fasttext_model()
nlp = load_spacy_model()
skillextractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
def preprocesstext(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens
def calculatesimilarity(doc1, doc2,doc1Text):
    
    
    similarity = fasttextmodel.wv.n_similarity(doc1, doc2)
    skillsData = skillextractor.annotate(doc1Text)
    return similarity,skillsData
def skillExtractor(tokens):
    extracTedData = []
    for token in tokens:
        extracTedData.append(skillextractor.annotate(token)) 
    return extracTedData
   
def extracttextfrompdf(pdfpath):
    text = ""
    with open(pdfpath, "rb") as pdffile:
        pdfreader = PyPDF2.PdfReader(pdffile)
        for page in pdfreader.pages:
            text += page.extract_text()
    return text

def resumeMatch(title,pdfpath,dess):
    #print("***Process of Resume Matching Started.***")
    
    def fileName(Type):
         
         imageextension= '.png'
         imagename = title + "" + Type
         todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
         newfilename = f"{imagename}{todaydate}{imageextension}"
         return newfilename
    def processtext(text):
        # Sentence tokenization
        sentences = sent_tokenize(text)
        
     
        #nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
        # Initialize NLTK's WordNetLemmatizer
        #nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        
        processedsentences = []
        for sentence in sentences:
            # Word tokenization
            words = word_tokenize(sentence)
            
            # Stop word removal and lemmatization
            filteredwords = [word.lower() for word in words if word.lower() not in stop_words]
            processedsentence = ' '.join(filteredwords)
            processedsentences.append(processedsentence)
        return processedsentences
   
    allDescriptions = []
    #print(dess)
    #print(dess[0])
    for row in dess:
        #jobDescriptionTokens = preprocesstext(row)
        ##print(row)
        allDescriptions.append(row)  # Add unique tokens to the set

    resumetext = extracttextfrompdf(pdfpath)
    resumetokens = (preprocesstext(resumetext))  # Create a set of unique resume tokens
  
    similarityscores = []
    c = 0
    descriptionSkills = []
    
    for description in allDescriptions:
        c += 1
      
        jobdescriptiontokens = preprocesstext(description)
        doc1 = preprocesstext(" ".join(jobdescriptiontokens))
        doc2 = preprocesstext(" ".join(resumetokens))
        similarity, skillsData = calculatesimilarity(doc1,doc2," ".join(jobdescriptiontokens) )
        #   #print(similarity)
        similarityscores.append(similarity)
        print(f"Similarity with Job Description {c}: {similarity:.2f}")
        descriptionSkills.append(skillsData)

        
        
  
    
    tokens2 = preprocesstext(" ".join(resumetokens))
    resumeSkills = skillExtractor(tokens2)
   
    return similarityscores,descriptionSkills
    #averagesimilarity = sum(similarityscores) / len(similarityscores)
    ##print(f"Average Similarity Score: {averagesimilarity:.2f}")
    
    # Display top 10 similarity scores using a bar graph
    """
    topscores = sorted(similarityscores, reverse=True)[:10]
    jobindices = sorted(range(len(similarityscores)), key=lambda k: similarityscores[k], reverse=True)[:10]
    joblabels = [f"Job {i+1}" for i in jobindices]
    plt.Figure(figsize=(13,13))
    plt.bar(joblabels, topscores)
    plt.xticks(rotation=90, ha='right')
    plt.xlabel(f'{title}Job Descriptions')
    plt.ylabel("Similarity Score")
    plt.title("Top 10 Similarity Scores")
    plt.annotate(f"Average Score on all descriptions: {averagesimilarity:.2f}", xy=(0.5, 0.95), xycoords="axes fraction", ha="center")
    plt.savefig("Plots/"+fileName("Resume Matching Score"))
    plt.show()
    """
"""
for skills1,skills2 isn zip(descriptionSkills,resummeskills):
    #print("Description Skills")

    
    for j in skills1[0]['results']['full_matches']:
                  
                        if j['score']==1  :
                            if j['doc_node_value'] not in skillsDesc:
                                skillsDesc.append( j['doc_node_value'])
                            
                    
    for t in skills1[0]['results']['ngram_scored']:
     
        if t['score']==1:
                
                    if t['doc_node_value'] not in skillsDesc:
                        
                       skillsDesc.append( t['doc_node_value'])
    #print(skillsDesc)
    #print("****")
    #print("Resume Skills")
    for j in skills2['results']['full_matches']:
                  
                        if j['score']==1  :
                            if j['doc_node_value'] not in skillsDesc:
                                skillsres.append( j['doc_node_value'])
                            
                    
    for t in skills2['results']['ngram_scored']:
     
        if t['score']==1:
                
                    if t['doc_node_value'] not in skillsDesc:
                        
                       skillsres.append( t['doc_node_value'])
    #print(skillsres)
"""