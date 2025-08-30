import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as pxs
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
# import matplotlib.pyplot as plt
@st.cache_data
def load_dt():
    data = pd.read_csv("src/Data/data_preprocess.csv")
    diseases = [i for i in data.columns if i.startswith("diseases_")]
    symptoms = [i for i in data.columns if not i.startswith("diseases_")]
    return data,diseases,symptoms
@st.cache_resource
def load_mdl():
    symp_rule = joblib.load(f"src\Models\FPgrowth.pkl")
    return symp_rule
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

data,diseases,symptoms = load_dt()
symp_rule = load_mdl()
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




st.header("Health-Care Model",divider=True)
st.markdown("<b>This model helps users identify potential diseases based on their current symptoms. It also recommends additional related symptoms to monitor and provides precautionary measures, using insights from past symptom patterns.</b>",unsafe_allow_html=True)
st.sidebar.subheader("Input Section")
st.sidebar.text("Choose atleast 2 symptoms.")
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
    s2 = st.sidebar.selectbox("Type Symptom 2",options=result_s1,index=None)


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
    s3 = st.sidebar.selectbox("Type Symptom 3",options=result_s2,index=None)

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
    s4 = st.sidebar.selectbox("Type Symptom 4",options=result_s3,index=None)





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
        avg = total/len(list(n_set))
        if int(avg)<=1:
            st.success("Symptoms are mild and common. Home care and rest are usually sufficient.")
            st.write("See Precautions for more details")
        elif int(avg)==2:
            st.warning("Symptoms are moderate. It is advisable to consult a doctor if they persist or worsen.")
            st.write("See Precautions for more details")
        elif int(avg)>=3:
            st.error("🚨 Symptoms are severe. Immediate medical attention or hospital visit is recommended.")
        pre_data = pd.read_csv("src/Data/Sources/symptoms_precautions_updated.csv")
        tfidf = TfidfVectorizer(stop_words='english') #removes stopwords (if,or,in etc)
        tx = tfidf.fit_transform(pre_data['Precaution'])
        s = list(n_set)
        string = " ".join(s)
        tr = tfidf.transform([string])
        sml = cosine_similarity(tr,tx)
        idx = sml.argmax()
        result_precautions = pre_data.loc[idx,["Precaution"]].values
        expander = st.expander("Show Precaution")
        expander.write(result_precautions[0])
        result_diseases = prediction_diseases(n_set)
        try:
          
           st.success("Possible Diseases and Symptoms")
          
           df1 = pd.DataFrame(result_diseases).sort_values(by="confidence",ascending=False)
           l = []
           l1 = []
           for i in df1['symptoms']:
               l.append(re.sub(f"[""''\[\]]+","",str(i)))
           df1['symptoms'] = l
           for i in df1['pos_diseases']:
               l1.append(re.sub(f"[""''\[\]]+","",str(i)))

           df1['pos_diseases']=l1 
           cols = st.columns(3)

           for j,i in enumerate(df1['symptoms'].value_counts().reset_index().head(3).values):
               new = df1[df1['symptoms']==i[0]].sort_values(by='confidence',ascending=False).head(5)
               fig = pxs.pie(new,values='confidence',names='pos_diseases',title=f"Possible Diseases with Symptom: {i[0]}")
               cols[j] = st.plotly_chart(fig,use_container_width=True)
            #    st.write(fig)
        except:
             df1 = pd.DataFrame(result_diseases).sort_values(by="confidence",ascending=False).head(3)
             df1["Diseases"] = df1["Diseases"].apply(lambda x: ", ".join(x) if isinstance(x, frozenset) else str(x)) #converting frozen type to list
             fig = pxs.pie(df1,values="confidence",names="Diseases")
             st.write(fig)

#--------------------------------------------------------------------------------
# result_s4 = prediction_symptoms(y2)
# s5 = st.sidebar.selectbox("Type Symptom 5",options=result_s4,accept_new_options=False)
# s4 = st.sidebar.selectbox("Type Symptom 4",options=set(symptoms)-{s1,s2,s3},accept_new_options=False,index=None)
# s5 = st.sidebar.selectbox("Type Symptom 5",options=set(symptoms)-{s1,s2,s3,s4},accept_new_options=False,index=None)
