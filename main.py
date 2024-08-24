from transformers import pipeline
import spacy
from spacy.matcher import Matcher
from flask import Flask, render_template, request
import os

app = Flask(__name__)

nlp_model = spacy.load('en_core_web_trf')  

pattern_matcher = Matcher(nlp_model.vocab)

# Define advanced patterns for legal clause and terminology detection
legal_patterns = [
    [{"LOWER": "whereas"}, {"IS_ALPHA": True, "OP": "*"}, {"LOWER": "therefore"}],  # Matches clauses starting with "Whereas" and ending with "Therefore"
    [{"ENT_TYPE": "LAW"}],  # Matches legal references
    [{"ENT_TYPE": "ORG"}, {"LOWER": "agrees"}, {"LOWER": "to"}],  # Matches organizational commitments
]

pattern_matcher.add("LEGAL_CLAUSE_PATTERNS", legal_patterns)

text_summarizer = pipeline("summarization", model="nlpaueb/legal-bert-base-uncased", tokenizer="nlpaueb/legal-bert-base-uncased")

UPLOAD_DIR = 'uploaded_files'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        uploaded_file = request.files.get('uploaded_file')

        if not uploaded_file or uploaded_file.filename == '':
            return render_template('index.html', error_msg='No file selected for upload.')

        if uploaded_file.filename.endswith('.txt'):
            safe_filename = os.path.basename(uploaded_file.filename)
            saved_file_path = os.path.join(UPLOAD_DIR, safe_filename)
            uploaded_file.save(saved_file_path)

            with open(saved_file_path, 'r', encoding='utf-8') as text_file:
                original_text = text_file.read()

            processed_doc = nlp_model(original_text)

            found_matches = pattern_matcher(processed_doc)
            extracted_phrases = [processed_doc[start:end].text for _, start, end in found_matches]

            summary_max_length = min(150, len(original_text.split()))

            try:
                generated_summary = text_summarizer(
                    original_text, max_length=summary_max_length, min_length=30, do_sample=False
                )
                summarized_text = generated_summary[0]['summary_text']
            except Exception as e:
                return render_template('index.html', error_msg=f"Error during summarization: {str(e)}")

            return render_template('index.html', 
                                   original_text=original_text, 
                                   extracted_phrases=extracted_phrases, 
                                   summarized_text=summarized_text)
        else:
            return render_template('index.html', error_msg='Invalid file type. Please upload a .txt file.')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
