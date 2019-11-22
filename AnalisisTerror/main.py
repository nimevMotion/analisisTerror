import spacy
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

sentimentData= pd.read_csv('senticnet5.txt',sep="\t",index_col=0)
nlp = spacy.load("en_core_web_md")
pos=["SYM","PUNCT","X","SPACE"]
books=[]
universo={}
dfHeatMap=pd.DataFrame(universo)

def main():
    global books
    global universo
    global dfHeatMap
    
    rutas=[]
    
    print("Numero de libros: ")
    tamBooks=input()
    
    for i in range(int(tamBooks)):
        print("Ruta: ")
        rutas.append(input())
        nomBook=obtenerNombre(rutas[i])
        books.append(nomBook)
        
    ind=crearIndex()
    dfHeatMap=pd.DataFrame(universo,index=ind)
    
    for i in range(int(tamBooks)):
        book=obtenerW(rutas[i])
        doc = nlp(book)
        heatmapPlot(doc,i)
        scatterPlot(doc,i)
        actDict(doc,i,tamBooks)
        print(dfHeatMap.head())
    
    #plt.figure(figsize=(50,5))
    sb.heatmap(dfHeatMap,center=0,xticklabels=False,cmap="coolwarm")
    plt.savefig("AnalizadorLibros")
       
def obtenerW(book):
    rawTxt = open(book,encoding='utf-8')
    text = rawTxt.read()
    return text

def obtenerNombre(ruta):
    posSeparador=ruta.rfind("\\")
    posUltima=ruta.rfind(".")
    nombre=ruta[posSeparador+1:posUltima]
    return nombre

def heatmapPlot(doc,i):
    global sentimentData
    global books
    global pos
    global universo
    
    coleccion={}
    nomPNG=books[i]+"_Heatmap"
           
    for token in doc:
        if (token.pos_ not in pos) & (token.text in sentimentData.index.values):
            coleccion.update({token.text:[sentimentData.at[token.text,"INTENSITY"]]})
                        
        bookData=pd.DataFrame(coleccion)
        
    plt.figure(figsize=(50,5))
    sb.heatmap(bookData,vmin=-1, vmax=1,xticklabels=False,yticklabels=False,center=0,cmap="coolwarm")
    plt.savefig(nomPNG)
    
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

def crearIndex():
    global universo
    global books
    aux=[]
    
    for i in range(len(books)):
        aux.append(books[i])
    return aux

def actDict(doc,i,tamBook):
    global dfHeatMap
    global books
    
    j=0
    vector=[0]
    
    for k in range(len(tamBook)):
        vector.insert(0,0)
        
    for token in doc:
        if (token.pos_ not in pos) & (token.text in sentimentData.index.values):
            if str(j) in dfHeatMap.head():
                dfHeatMap.loc[books[i],str(j)]=sentimentData.at[token.text,"INTENSITY"]
            else:
                for k in range(len(tamBook)):
                    vector[k]=0
                vector[i]=sentimentData.at[token.text,"INTENSITY"]
                dfHeatMap[str(j)] = vector
            j=j+1
        
if __name__ == "__main__":
    main()    

    
    

    


