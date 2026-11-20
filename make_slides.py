from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR

prs=Presentation(); prs.slide_width= Inches(13.333); prs.slide_height= Inches(7.5)
NAVY=RGBColor(20,35,60); BLUE=RGBColor(45,105,180); PALE=RGBColor(235,242,250); DARK=RGBColor(45,52,62); WHITE=RGBColor(255,255,255); TEAL=RGBColor(35,145,135)

def base(title, subtitle=None):
    s=prs.slides.add_slide(prs.slide_layouts[6]); bg=s.background.fill; bg.solid(); bg.fore_color.rgb=WHITE
    bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width, Inches(.18)); bar.fill.solid(); bar.fill.fore_color.rgb=BLUE; bar.line.fill.background()
    tx=s.shapes.add_textbox(Inches(.65), Inches(.45), Inches(12), Inches(.55)).text_frame; tx.text=title; tx.paragraphs[0].font.size=Pt(27); tx.paragraphs[0].font.bold=True; tx.paragraphs[0].font.color.rgb=NAVY
    if subtitle:
        p=s.shapes.add_textbox(Inches(.68), Inches(1.06), Inches(12), Inches(.35)).text_frame.paragraphs[0]; p.text=subtitle; p.font.size=Pt(11); p.font.color.rgb=RGBColor(105,115,130)
    return s

def textbox(s,x,y,w,h,text,size=17,bold=False,color=DARK):
    tf=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap=True; tf.text=text
    for p in tf.paragraphs: p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=color
    return tf

def bullets(s, items, x=.9,y=1.65,w=11.5,h=5.2,size=19):
    tf=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap=True; tf.clear()
    for i,item in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=item; p.level=0; p.font.size=Pt(size); p.font.color.rgb=DARK; p.space_after=Pt(13)

def box(s,x,y,w,h,title,body,fill=PALE):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); sh.fill.solid(); sh.fill.fore_color.rgb=fill; sh.line.color.rgb=RGBColor(210,220,232)
    tf=sh.text_frame; tf.clear(); tf.margin_left=Inches(.15); tf.margin_top=Inches(.12)
    p=tf.paragraphs[0]; p.text=title; p.font.bold=True; p.font.size=Pt(15); p.font.color.rgb=NAVY
    p=tf.add_paragraph(); p.text=body; p.font.size=Pt(11); p.font.color.rgb=DARK

# 1
s=base('SmartSupport AI','SWE015 – Introduction to Large Language Models | Individual Homework')
textbox(s,.85,1.65,11.6,.75,'Bilingual customer-support ticket routing with a four-model comparison',25,True,NAVY)
box(s,.9,3.0,5.5,2.1,'Student details','Name: [YOUR FULL NAME]\nStudent ID: [YOUR STUDENT ID]\nUse case: 2 – Customer Support Ticket Router',PALE)
box(s,6.8,3.0,5.5,2.1,'Submission links','Repository: [ADD YOUR PUBLIC/SHARED GITHUB LINK]\nDemo video: [ADD UNLISTED VIDEO LINK]\nDataset: Amazon MASSIVE',PALE)
textbox(s,.9,6.25,11.5,.4,'Replace all bracketed fields before submission. Add links only after testing them in a private browser.',12,False,RGBColor(150,70,55))
# 2
s=base('Introduction – the problem','A real-world routing problem for appliance customer support')
bullets(s,["Support teams receive high volumes of short, noisy messages in Turkish and English.","Manual triage delays resolution and can send a ticket to the wrong team.","SmartSupport AI predicts the customer intent, estimates confidence, and routes the ticket.","Low-confidence or unsupported messages are escalated to a human agent; greetings receive a rule-based response.","Success means accurate intent classification, useful escalation, and low inference latency."],size=18)
# 3
s=base('Data and preprocessing','Dataset source, splits and normalization')
box(s,.8,1.65,3.8,2.1,'Source','Amazon MASSIVE\nLocales: en-US and tr-TR\nPublic multilingual NLU dataset')
box(s,4.8,1.65,3.8,2.1,'Partitions','Use dataset-provided train, validation and test partitions.\nExact row counts: [FILL FROM YOUR RUN]')
box(s,8.8,1.65,3.8,2.1,'Preprocessing','Unicode normalization, lowercase, repeated-letter cleanup, punctuation cleanup, model-number masking, whitespace normalization.')
bullets(s,["Keep the same held-out test partition for all four models.","Map intent labels to stable string names and report the exact split sizes.","Cite the MASSIVE dataset card and check its terms before redistribution."],x=1,y=4.35,w=11.2,h=2.2,size=16)
# 4
s=base('System architecture','Data flow from user input to the production quantized model')
box(s,.55,1.8,2.15,1.25,'User input','Streamlit text box')
box(s,3.0,1.8,2.15,1.25,'Preprocessing','Regex + Unicode normalization')
box(s,5.45,1.15,2.2,1.25,'Model comparison','TF-IDF / NN / mBERT')
box(s,5.45,3.0,2.2,1.25,'Production model','Distilled DistilBERT INT8')
box(s,8.0,1.8,2.0,1.25,'Post-process','Confidence threshold + team map')
box(s,10.35,1.8,2.35,1.25,'Output','Intent, confidence, team / human')
for x1,y1,x2,y2 in [(2.7,2.42,3,2.42),(5.15,2.42,5.45,1.78),(5.15,2.42,5.45,3.62),(7.65,1.78,8,2.42),(7.65,3.62,8,2.42),(10,2.42,10.35,2.42)]:
    line=s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)); line.line.color.rgb=BLUE; line.line.width=Pt(2)
textbox(s,.8,5.25,11.7,.8,'Runs locally on the demo machine. The Streamlit application uses the INT8 quantized Transformer for its final routing decision.',16,False,NAVY)
# 5
s=base('Technique map','At least one technique from each required area')
box(s,.7,1.65,5.8,1.65,'A · Neural networks & deep learning','Word2Vec averaged embeddings → feedforward neural network; trained with SGD and backpropagation.')
box(s,6.8,1.65,5.8,1.65,'B · Classical NLP','Regex normalization, model-code masking, fuzzy edit-distance-style correction helper, ELIZA-style small-talk rules, TF-IDF + Logistic Regression.')
box(s,.7,3.65,5.8,1.65,'C · Transformers','Fine-tuned multilingual BERT for intent classification; multilingual subword tokenizer.')
box(s,6.8,3.65,5.8,1.65,'D · Efficient models','Knowledge distillation into DistilBERT followed by dynamic INT8 quantization; model size and latency benchmarking.')
# 6
s=base('Four-model ladder','All models use the same held-out test set')
box(s,.7,1.75,2.85,2.2,'1 · Classical baseline','TF-IDF n-grams + Logistic Regression\nFast, interpretable baseline')
box(s,3.75,1.75,2.85,2.2,'2 · Neural network','Word2Vec averaged vectors + MLP\nTrained with SGD/backprop')
box(s,6.8,1.75,2.85,2.2,'3 · Transformer','Fine-tuned bert-base-multilingual-cased\nTeacher model')
box(s,9.85,1.75,2.85,2.2,'4 · Quantized Transformer','Distilled multilingual DistilBERT + dynamic INT8\nUsed by the application',RGBColor(228,245,242))
bullets(s,["Training settings and checkpoint details are recorded by the training script.","Report actual outcomes from your own run; a Transformer need not beat the baseline to be a valid finding."],x=1,y=4.6,w=11.2,h=1.6,size=16)
# 7 results
s=base('Results – measured comparison','Populate this table only from models/results.csv after a successful run')
rows=5; cols=5; table=s.shapes.add_table(rows,cols, Inches(.65), Inches(1.8), Inches(12), Inches(3.1)).table
headers=['Model','Macro-F1','Accuracy','Size (MB)','Latency (ms)']
for j,h in enumerate(headers): table.cell(0,j).text=h
models=['TF-IDF + Logistic Regression','Word2Vec + Feedforward NN','mBERT fine-tuned','Distilled Transformer INT8']
for i,m in enumerate(models,1):
    table.cell(i,0).text=m
    for j in range(1,5): table.cell(i,j).text='[FILL FROM results.csv]'
for row in table.rows:
    for cell in row.cells:
        for p in cell.text_frame.paragraphs:
            p.font.size=Pt(11); p.font.color.rgb=DARK
textbox(s,.85,5.35,11.5,.8,'Hardware: [CPU/GPU model, RAM, OS]. Latency must be averaged over at least 100 runs. Use the exact measured values; do not copy numbers from papers or AI.',15,False,RGBColor(150,70,55))
# 8
s=base('Application and live behavior','The quantized Transformer powers the final routing decision')
bullets(s,["Interactive Streamlit interface accepts a new free-text support message.","Four model predictions, confidence scores and per-request latency are shown side by side.","The INT8 quantized Transformer determines the final team assignment.","If confidence is below the user-selected threshold, route to Human Agent.","Greetings and small talk receive a rule-based response instead of a ticket classification."],size=18)
# 9
s=base('Conclusion, limitations and next steps','Complete this slide after testing the application')
box(s,.75,1.65,3.8,2.15,'What the comparison showed','[Describe which model performed best and which was fastest using your measured results.]')
box(s,4.8,1.65,3.8,2.15,'Failure example','[Paste one real test message the system misclassified; explain why it failed.]')
box(s,8.85,1.65,3.8,2.15,'Limitations / next steps','MASSIVE is general voice-assistant intent data, not appliance repair tickets. Confidence threshold is an abstention proxy, not a trained OOS class. Add domain examples and test calibration.')
bullets(s,["A modest score is acceptable if the experiment is measured honestly.","Next: add real appliance-specific examples, improve confidence calibration, and test CPU latency."],x=1,y=4.45,w=11.2,h=1.6,size=16)
# 10
s=base('AI usage and references','Required disclosure')
bullets(s,["AI usage: ChatGPT assisted with drafting the project scaffold, code, README, presentation outline and demo narration. I ran, tested, debugged and reviewed the submitted implementation and am responsible for understanding it.","Dataset: Amazon Science, MASSIVE – https://huggingface.co/datasets/AmazonScience/massive", "Model references: bert-base-multilingual-cased and distilbert-base-multilingual-cased model cards on Hugging Face.","Before submission: add your name, student ID, repository URL, demo URL, exact split sizes, measured results, hardware, and one real failure example."],size=16)

prs.save('/mnt/data/smartsupport_ai/SmartSupport_AI_Presentation_DRAFT.pptx')
print('created presentation')
