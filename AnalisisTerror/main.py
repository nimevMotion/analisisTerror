import spacy
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random

sentimentData= pd.read_csv('senticnet5.txt',sep="\t",index_col=0)
affectIntensityData=pd.read_csv('NRC_AffectIntensity-Lexicon.txt',sep="\t",index_col=0)
nlp = spacy.load("en_core_web_md")
pos=["SYM","PUNCT","X","SPACE"]
books=[]
datos={"book":"","tokens":"","intensity":"","anger":"","joy":"","sadness":"","fear":"","tam_intensity":"","tam_anger":"","tam_joy":"","tam_sadness":"","tam_fear":""}

def main():
    global books
    global universo
    global dfHeatMap
    global datos
    
    rutas=[]
    
    print("Numero de libros: ")
    tamBooks=input()
    
    for i in range(int(tamBooks)):
        print("Ruta: ")
        rutas.append(input())
        nomBook=obtenerNombre(rutas[i])
        books.append(nomBook)
   
    for i in range(int(tamBooks)):
        book=obtenerW(rutas[i])
        doc = nlp(book)
        heatmapPlot(doc,i)
        scatterPlot(doc,i)
        sentimentPlot(doc,i)
        print(datos)
        
        if datos["tam_intensity"]>0:
            print("El libro "+books[i]+" es POSITIVO")
        else:
            print("El libro "+books[i]+" es NEGATIVO") 

#================================================================================================
#La función obtenerW obtiene el texto sin procesar
#================================================================================================
def obtenerW(book):
    rawTxt = open(book,encoding='utf-8')
    text = rawTxt.read()
    return text

#================================================================================================
#La función obtener nombre, regresa el nombre del libro para identificar la gráficas
#================================================================================================
def obtenerNombre(ruta):
    posSeparador=ruta.rfind("\\")
    posUltima=ruta.rfind(".")
    nombre=ruta[posSeparador+1:posUltima]
    return nombre

#================================================================================================
#La función heatmapPlot dibuja el mapa de calor de la polaridad de las palabras
#================================================================================================
def heatmapPlot(doc,i):
    global sentimentData
    global books
    global pos
    global datos
    
    coleccion={}
    nomPNG=books[i]+"_Heatmap"
    datos["book"]=books[i]
    polaridad=0
    tampolaridad=0
    totTokens=0
           
    for token in doc:
        if (token.pos_ not in pos) & (token.text in sentimentData.index.values):
            coleccion.update({token.text:[sentimentData.at[token.text,"INTENSITY"]]})
            polaridad=polaridad+float(sentimentData.at[token.text,"INTENSITY"])
            tampolaridad=tampolaridad+1
                        
        bookData=pd.DataFrame(coleccion)
        totTokens=totTokens+1
        
    datos["intensity"]=polaridad
    datos["tam_intensity"]=tampolaridad
    datos["tokens"]=totTokens
    plt.figure(figsize=(50,5))
    sb.heatmap(bookData,vmin=-1, vmax=1,xticklabels=False,yticklabels=False,center=0,cmap="coolwarm").set_title(nomPNG)
    plt.savefig(nomPNG)

#================================================================================================
#La función sentimentPlotr, dibuja la gráfica con el desglose de emociones
#================================================================================================
def sentimentPlot(doc,i):
    global affectIntensityData
    global sentimentData
    global books
    global pos
    global datos
    
    nomPNG=books[i]+"_SentimentHeatmap"
    score=[]
    affect=[]
    joy=0
    fear=0
    sadness=0
    anger=0
    tamjoy=0
    tamfear=0
    tamsadness=0
    tamanger=0
    
    for token in doc:
        if (token.pos_ not in pos):
            if(token.text in affectIntensityData.index.values):
                auxScore=affectIntensityData.at[token.text,"score"]
                auxAffect=affectIntensityData.at[token.text,"AffectDimension"]
                if isinstance(auxAffect, str):
                    score.append(float(affectIntensityData.at[token.text,"score"]))
                    affect.append(str(affectIntensityData.at[token.text,"AffectDimension"]))
                    
                else:
                    auxScore=auxScore.tolist()
                    auxAffect=auxAffect.tolist()
                    i=random.randint(0, len(auxScore)-1)
                    score.append(float(auxScore[i]))
                    affect.append(str(auxAffect[i]))
                    auxScore.clear()
                    auxAffect.clear()
    
    for x in range(len(score)):
        if affect[x] == "joy":
            joy=joy+score[x]
            tamjoy=tamjoy+1
        elif affect[x] == "fear":
            fear=fear+score[x]
            tamfear=tamfear+1
        elif affect[x] == "anger":
            anger=anger+score[x]
            tamanger=tamanger+1
        elif affect[x] == "sadness":
            sadness=sadness+score[x]
            tamsadness=tamsadness+1
    
    datos["joy"]=joy
    datos["tam_joy"]=tamjoy 
    datos["fear"]=fear
    datos["tam_fear"]=tamfear
    datos["anger"]=anger
    datos["tam_anger"]=tamanger
    datos["sadness"]=sadness
    datos["tam_sadness"]=tamsadness
    data={"SCORE":score,"AFFECT":affect}   
    bookD=pd.DataFrame(data)
    sb.set(style='darkgrid')
    try:
        sb.scatterplot(y = "SCORE", x = bookD.index.values,hue="AFFECT", data=bookD)
    except(RuntimeError, TypeError, NameError) as e:
        print(e)    
    plt.savefig(nomPNG)
    
#================================================================================================
#La función scatterPlot dibuja la polaridad del texto
#================================================================================================
def scatterPlot(doc,i):
    global sentimentData
    global books
    global pos
    
    coleccion={}
    nomPNG=books[i]+"_Scatterplot"
    bookData=pd.DataFrame(coleccion)
        
    for token in doc:
        if (token.pos_ not in pos) & (token.text in sentimentData.index.values):
            aux={'INTENSITY':sentimentData.at[token.text,"INTENSITY"]}
            bookData = bookData.append(aux, ignore_index=True)
            
    plt.figure(figsize=(50,5))
    sb.set(style='darkgrid')
    sb.scatterplot(x = bookData.index.values, y = "INTENSITY", data=bookData)
    plt.savefig(nomPNG)

if __name__ == "__main__":
    main()
