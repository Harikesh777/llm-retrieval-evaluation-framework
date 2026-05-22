import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shutil

DATA_DIR = "data/raw_docs"
EVAL_FILE = "data/evaluation_dataset.json"

if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

dataset = []

for i in range(1, 21):
    filename = f"document_{i:02d}.pdf"
    filepath = os.path.join(DATA_DIR, filename)
    
    company_name = f"Company {i} Alpha"
    revenue = f"${i * 10 + 5} Million"
    secret_code = f"Project Omega-{i * 100}"
    
    page1_text = f"Annual Report for {company_name}\n\n" * 5 + f"This year, {company_name} reported a total revenue of {revenue}.\nWe are extremely proud of this growth.\n\n" * 5
    page2_text = f"Future Outlook and Internal Strategies\n\n" * 5 + f"Our main focus for next year is the secretive {secret_code}, which will revolutionize the industry.\n\n" * 5
    
    with PdfPages(filepath) as pdf:
        fig = plt.figure(figsize=(8, 11))
        fig.text(0.1, 0.9, page1_text, fontsize=12, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        
        fig = plt.figure(figsize=(8, 11))
        fig.text(0.1, 0.9, page2_text, fontsize=12, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        
    print(f"Created {filename}")
    
    question = f"What was the total revenue reported by {company_name}?"
    answer = f"{company_name} reported a total revenue of {revenue}"
    
    dataset.append({
        "question": question,
        "answer": answer,
        "source": filename
    })

with open(EVAL_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Evaluation dataset saved to {EVAL_FILE} with {len(dataset)} entries.")
