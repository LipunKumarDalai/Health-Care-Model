import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as pxs
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image
import json

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# import matplotlib.pyplot as plt
@st.cache_data
def load_dt():
    data = pd.read_csv("src/Data/data_preprocess.csv")
    data_c = pd.read_csv("src/Data/LblEncodeDF.csv")
    diseases = [i for i in data.columns if i.startswith("diseases_")]
    symptoms = [i for i in data.columns if not i.startswith("diseases_")]
    return data,diseases,symptoms,data_c
@st.cache_resource
def load_mdl():
    symp_rule = joblib.load(f"src\Models\FPgrowth1.pkl")
    disease_neural = joblib.load("src/Models/Neural.pkl")
    label_ = joblib.load("src/Models/LabelEncode.pkl")
    return symp_rule,disease_neural,label_
@st.cache_data
def load_rules(symp_rule,symptoms,diseases):
    symptom = symp_rule[
        symp_rule['antecedents'].apply(lambda x: all(i in symptoms for i in x))&
        symp_rule['consequents'].apply(lambda x: all(i in symptoms for i in x))
        ]
    disease = symp_rule[
        symp_rule["antecedents"].apply(lambda x: all(i in symptoms for i in x))&
        symp_rule["consequents"].apply(lambda x: all(i in diseases for i in x))
    ]
    return symptom,disease

data,diseases,symptoms,data_ac = load_dt()
symp_rule,disease_neural,label_ = load_mdl()
symptom,disease =load_rules(symp_rule,symptoms,diseases)

def jaccard_sml(s1,s2):
    s1 = set(s1)
    s2 = set(s2)
    itrsec = len(s1.intersection(s2))
    union = len(s1.union(s2))
    sml = itrsec / union if union !=0 else 0
    return sml

def prediction_symptoms(user_symp):
    result = []
    jc_ = []
    for idx,row in symptom.iterrows():
        antecedents = row["antecedents"]
        consequents = row["consequents"]
        confidence = row["confidence"]
        jc = jaccard_sml(user_symp,antecedents)
        jc_.append(jc)
        if jc > 0.01 and jc!=1.0:
            result.append({
                "symptoms":list(antecedents-user_symp),
                "confidence":confidence,
            })
        elif jc==1.0:
            result.append({
                "symptoms":list(consequents),
                "confidence":confidence,
            })
        # if antecedents.issubset(user_symp) and confidence > 0.5:
        #    result.append([consequents,confidence])
    output_set = set()
    if len(result)>=1:
        output_df = pd.DataFrame(result).sort_values(by="confidence",ascending=False)["symptoms"].values
        for i in output_df:
            output_set.add(str(i).strip("[]"))
    else:
        pass
    return output_set

def prediction_diseases(user_symp):
    result_actual = []
    result_missing = []
    for idx,row in disease.iterrows():
        antecedents = row["antecedents"]
        consequents = row["consequents"]
        confidence = row["confidence"]
        jc = jaccard_sml(user_symp,antecedents)
        if jc > 0.1 and jc!=1.0:
            result_missing.append({
                "symptoms":list(antecedents-user_symp),
                "pos_diseases":list(consequents),
                "confidence":confidence,
            })
        elif jc==1.0:
            result_actual.append({
                "Diseases":consequents,
                "confidence":confidence
            })
            
       # if antecedents.issubset(user_symp) and confidence > 0.5:
       #    result.append([consequents,confidence])
    if len(result_actual)>=1:
        return result_actual
    else:
        return result_missing




# st.write("<h1 style='color:white'>Health-Care Model</h1>",unsafe_allow_html=True)
# st.write("<b style='color:white'>This model helps users identify potential diseases based on their current symptoms. It also recommends additional related symptoms to monitor and provides precautionary measures, using insights from past symptom patterns.</b>",unsafe_allow_html=True)
st.sidebar.subheader("Input Section")
st.sidebar.text("Choose atleast 2 symptoms.")
st.write("<h1 style='color:black;text-align:center'>Well Predict</h1>",unsafe_allow_html=True)

img = Image.open("image/health.jpg")
img = img.resize((900, 600))

c1,c2 = st.columns([0.6,0.4])
with c1:
    st.write("<p><b style=color:black;>Our Well Predict prediction system is designed to assist users in identifying possible diseases based on the symptoms they provide. If a user enters all symptoms correctly, the model evaluates the full combination to generate the most accurate disease prediction. However, in real scenarios users may sometimes enter incomplete or incorrect symptoms. In such cases, the system is capable of adapting intelligently:</b></p>" \
    "<p><b  style=color:black;>Incorrect or unrelated inputs → The model analyzes available symptoms and attempts to match at least one symptom with known conditions.</b></p>" \
    "<p><b style=color:black;>Partial matches → If only one or two valid symptoms are detected, the system still predicts diseases associated with those symptoms.</b></p>" \
    "<p><b style=color:black;>Fallback approach → Even with a single valid symptom, the model provides a list of the most likely diseases linked to that symptom.</b></p>" \
    "<p><b style=color:black;>In addition to disease predictions, the system also suggests related symptoms that are commonly observed together, helping users cross-check and refine their input. This ensures that the tool remains helpful and provides meaningful insights even when the input data is incomplete, noisy, or uncertain. The ultimate goal is to support early guidance and awareness, not to replace professional medical advice. </b></p>",unsafe_allow_html=True)
with c2:
    st.image(img)
#SYMPTOMS RECOMMENDATION
#--------------------------------------------------------------------------------
s1 = st.sidebar.selectbox("Type Symptom 1",options=symptoms,index=None)
result_s1 = prediction_symptoms({s1})
y = set()
if result_s1:
    s2 = st.sidebar.selectbox("Type Symptom 2",options=list(result_s1)+["other"],index=None)
    if s2=="other":
        s2 = st.sidebar.selectbox("Choose other Symptom",options=symptoms,width=300,key=0)
    try:
       reg = re.sub(f"[""'']+","",s2)
       y = set(reg.split(","))
       y.add(s1)
    except:
        pass  
else:
    s2 = st.sidebar.selectbox("Type Symptom 2",options=set(symptoms)-{s1},index=None)
    if s2:
        y.add(s1)
        y.add(s2)



result_s2 = prediction_symptoms(y)
y2 = set()
if result_s2:
    s3 = st.sidebar.selectbox("Type Symptom 3",options=list(result_s2)+["other"],index=None)
    if s3=="other":
        s3 = st.sidebar.selectbox("Choose other Symptom",options=symptoms,width=300,key=1)
    try:
        reg2 = re.sub(f"[""'']+","",s3)
        y2 = set(reg2.split(","))
        y2.add(s2)
    except:
        pass
else:
    s3 = st.sidebar.selectbox("Type Symptom 3",options=set(symptoms)-result_s2,index=None)
    if s3:
        y2.add(s3)

result_s3 = prediction_symptoms(y2)
y_final = set()
if result_s3:
    s4 = st.sidebar.selectbox("Type Symptom 4",options=list(result_s3)+["other"],index=None)
    if s4=="other":
        s4 = st.sidebar.selectbox("Choose other Symptom",options=symptoms,width=300,key=2)
    try:
        reg3 = re.sub(f"[""'']+","",s4)
        y_final = set(reg3.split(","))
        y_final.add(s3)
    except:
        pass
    
else:
    s4 = st.sidebar.selectbox("Type Symptom 4",options=set(symptoms)-result_s3,index=None)
    if s4:
        y_final.add(s4)


#DISEASES PREDICTION
#--------------------------------------------------------------------------------
y_list = list(y)+list(y2)+list(y_final)
n = []
for i in list(y_list):
    d = i.split(",")
    if len(d)>1:
        for j in d:
            n.append(re.sub(f"[''""]+","",j))
    else:
       n.append(re.sub(f"[''""]+","",i))
n_set = set(n)

if s1 and s2:
    submit = st.sidebar.button("Submit")
    if submit:
        svr = pd.read_csv("src/Data/SYMPTOMS_SEVERE.csv")
        total = 0
        for i,j in enumerate(svr["Symptoms"]):
            if j in list(n_set):
               indx = i
               total += svr.loc[indx,["Severity_level"]]
        if len(list(n_set))!=0:
            avg = total/len(list(n_set))
            if int(avg)<=1:
               st.success("Symptoms are mild and common. Home care and rest are usually sufficient.")
            elif int(avg)==2:
               st.warning("Symptoms are moderate. It is advisable to consult a doctor if they persist or worsen.")
            elif int(avg)>=3:
               st.error("🚨 Symptoms are severe. Immediate medical attention or hospital visit is recommended.")
            

            d_lst = list(n_set)
            inputs = np.zeros(377,dtype="int")
            for i,j in enumerate(data_ac.columns[1:]):
               if j in d_lst:
                  inputs[i] = 1
            result = disease_neural.predict(inputs.reshape(1,-1))
            maxi_label = np.argmax(result)
            out = label_.inverse_transform([maxi_label])
            st.success("Disease Predicted",width=180)
            st.write("<b style='color:black'>Disease Predicted:</b>",f"<b style='color:black'>{''.join(out)}</b>",unsafe_allow_html=True)



    
            pre_data = pd.read_csv("src/Data/Sources/symptoms_precautions_updated.csv")
            tfidf = TfidfVectorizer(stop_words='english') #removes stopwords (if,or,in etc)
            tx = tfidf.fit_transform(pre_data['Precaution'])
            s = list(n_set)
            string = " ".join(s)
            tr = tfidf.transform([string])
            sml = cosine_similarity(tr,tx)
            idx = sml.argmax()
            result_precautions = pre_data.loc[idx,["Precaution"]].values
            expander = st.expander("Quick Precaution")
            expander.write(f"<b style='color:black'>{result_precautions[0]}</b>",unsafe_allow_html=True)
            result_diseases = prediction_diseases(n_set)
            try:
          
               
          
               df1 = pd.DataFrame(result_diseases).sort_values(by="confidence",ascending=False)
               l = []
               l1 = []
               for i in df1['symptoms']:
                   l.append(re.sub(f"[""''\[\]]+","",str(i)))
               df1['symptoms'] = l
               for i in df1['pos_diseases']:
                  l1.append(re.sub(f"[""''\[\]]+","",str(i)))
               df1['pos_diseases']=l1 
               csv = df1.drop(columns="confidence")
               csv_dt = csv.to_csv(index=False)
               st.sidebar.text("Find More Possible diseases with symptoms")
               st.sidebar.download_button("Download csv",csv_dt,file_name="Diseases_symptoms.csv")
               st.success("Possible Diseases and Symptoms You Might Face.")
               cols = st.columns([0.33,0.33,.33])
               for j,i in enumerate(df1['symptoms'].value_counts().reset_index().head(3).values):
                  new = df1[df1['symptoms']==i[0]].sort_values(by='confidence',ascending=False).head(5)
                  fig = pxs.pie(new,values='confidence',names='pos_diseases',title=f"Possible Symptom: {i[0]}")
                  with cols[j]: st.plotly_chart(fig,use_container_width=True)
            #    st.write(fig)
            except:
                st.warning("No additional Symptoms and diseases found")

                # df1 = pd.DataFrame(result_diseases).sort_values(by="confidence",ascending=False).head(3)
                # df1["Diseases"] = df1["Diseases"].apply(lambda x: ", ".join(x) if isinstance(x, frozenset) else str(x)) #converting frozen type to list
                # fig = pxs.pie(df1,values="confidence",names="Diseases")
                # st.write(fig)


            #detailed precautions
            with open("src/Data/P_m.json","r") as rd: 
                med = json.load(rd)
            medicine = []
            detail_pre = []
            for i in med:
                if i["symptom"] in s:
                       detail_pre.append(i["precautions"])
                       medicine.append([i["symptom"],i["medicines"]])
            st.success("Precautions Predicted")
            for i in detail_pre:
                st.write(f"<b>{' '.join(i)}\n</b>",unsafe_allow_html=True)
            st.success("Medicines Predicted")
            for i,j in enumerate(medicine):
                st.write(f"<b>{j[0] +'→'}</b>",unsafe_allow_html=True)
                for k in j[1]:
                    st.write(f"<b>{k['name']+' : '+k['dose']}</b>",unsafe_allow_html=True)
                st.write(" ")




        else:
            st.error("🚨We couldn’t find any matches for the symptoms you entered.Its an issue from our side.Please try different symptom combinations.")

            
        

