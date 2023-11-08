from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from wordcloud import WordCloud
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from SkillPlotter import skillPlotter
import shutil
import ResumeMatch
import os
import matplotlib.pyplot as plt
app = FastAPI()

IMAGE_FOLDER = 'images'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Ensure the directory exists
os.makedirs(IMAGE_FOLDER, exist_ok=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DescriptionInput(BaseModel):
    descriptions: list[str]




@app.post("/process_descriptions")
async def process_description( desc_input: DescriptionInput):
    
  
    #print("here")
    scores,descriptionSkills = ResumeMatch.resumeMatch("software Developer","temp.pdf",desc_input.descriptions)
    ##print(descriptionSkills)
    os.remove("temp.pdf")
    newSkills =skillPlotter(descriptionSkills)
    #print("##########")
    #print(newSkills)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(newSkills)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    image_path = os.path.join(IMAGE_FOLDER, 'Skills.png')
    plt.savefig(image_path)
  
    return {"message": [float(score) for score in scores]}

@app.get('/getImage')
async def get_wordcloud_image():
    image_path = os.path.join(IMAGE_FOLDER, 'Skills.png')
    return FileResponse(image_path)


@app.post('/GetFile')
async def getFile(pdfFile: UploadFile):
     with open("temp.pdf", "wb") as temp_file:
        shutil.copyfileobj(pdfFile.file, temp_file)